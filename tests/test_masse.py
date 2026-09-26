"""« Compressible » est une conclusion sur le fluide, pas sur le champ.

Le corrige passait de div V != 0 a « l'ecoulement est compressible » sans
le maillon qui autorise ce pas : la conservation de la masse d'une particule
(cours III.6 -- un systeme a masse constante et volume variable). La loi de
masse volumique qui en decoule n'etait jamais ecrite.
"""

import pytest
import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.masse import (
    loi_de_masse_volumique,
    masse_de_la_particule,
    taux_de_variation_de_la_masse_volumique,
)
from polytech.fluides.domain.cinematique.symboles import T


def test_la_masse_volumique_est_l_inverse_du_volume() -> None:
    """M = rho V constante : si le volume double, la masse volumique halve."""
    divergence = enonce.CHAMP_B.divergence()
    produit = loi_de_masse_volumique(divergence) * enonce.CHAMP_B.loi_de_volume()
    assert sp.simplify(produit - 1) == 0


def test_la_masse_de_la_particule_reste_constante() -> None:
    """C'est l'enonce meme de III.6, verifie sur l'ecoulement compressible."""
    assert masse_de_la_particule(enonce.CHAMP_B.divergence()) == 1


def test_la_masse_volumique_de_b_croit_apres_une_seconde() -> None:
    """Div V > 0 pour t > 1 : le volume enfle, donc la densite chute."""
    loi = loi_de_masse_volumique(enonce.CHAMP_B.divergence())
    assert float(loi.subs(T, 2.0)) < float(loi.subs(T, 1.0))


def test_la_forme_differentielle_est_l_equation_de_continuite() -> None:
    """(1/rho) Drho/Dt = - div V : la conservation de la masse, en local."""
    divergence = enonce.CHAMP_B.divergence()
    assert (
        sp.simplify(
            taux_de_variation_de_la_masse_volumique(divergence) + divergence,
        )
        == 0
    )


def test_un_ecoulement_incompressible_garde_sa_masse_volumique() -> None:
    """Div V = 0 : la particule ne change ni de volume ni de densite."""
    loi = loi_de_masse_volumique(enonce.CHAMP_C.divergence())
    assert sp.simplify(loi - 1) == 0


def test_une_divergence_sans_primitive_close_est_refusee() -> None:
    """Sans primitive, pas de loi : on leve plutot que de rendre une integrale.

    Une divergence laissee abstraite est le cas le plus net : sympy sait
    integrer exp(t^2) (via erfi), mais pas une fonction quelconque.
    """
    with pytest.raises(ValueError, match="primitive"):
        loi_de_masse_volumique(sp.Function("d")(T))
