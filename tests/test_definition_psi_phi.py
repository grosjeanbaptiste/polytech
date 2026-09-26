"""psi et phi doivent verifier leur *definition*, pas une reponse memorisee.

Les tests existants comparent `phi` a `x^2 - y^2` : ils confrontent le
calcul a ce que l'on savait deja. Si la reconstruction par recollement
produisait une primitive plausible mais fausse sur un champ inconnu, rien
ne l'attraperait. Ces tests verifient la propriete qui *definit* les deux
fonctions -- et sur un champ dont la reponse n'a pas ete apprise par coeur.
"""

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.symboles import T, X, Y
from polytech.fluides.domain.cinematique.validite import (
    residu_de_la_fonction_de_courant,
    residu_du_potentiel,
)

TEMOIN = ChampDeVitesse(u=3 * X + Y, v=X - 3 * Y)
"""Champ arbitraire, a divergence nulle et irrotationnel : psi et phi y existent.

Sa fonction de courant et son potentiel n'ont pas ete calcules a la main :
c'est tout l'interet. Seule leur definition est verifiee.
"""


def test_le_temoin_admet_bien_les_deux_fonctions() -> None:
    """Sans cela le test suivant serait vide de sens."""
    assert TEMOIN.est_incompressible() is True
    assert TEMOIN.est_rotationnel() is False


def test_le_potentiel_de_c_engendre_bien_le_champ() -> None:
    """grad(phi) = V : c'est la definition, non la valeur attendue."""
    assert residu_du_potentiel(enonce.CHAMP_C) == (0, 0)


def test_la_fonction_de_courant_de_c_engendre_bien_le_champ() -> None:
    """U = dpsi/dy et v = -dpsi/dx."""
    assert residu_de_la_fonction_de_courant(enonce.CHAMP_C) == (0, 0)


def test_le_potentiel_du_temoin_engendre_son_champ() -> None:
    """La construction vaut donc en general, pas seulement sur le cas connu."""
    assert residu_du_potentiel(TEMOIN) == (0, 0)


def test_la_fonction_de_courant_du_temoin_engendre_son_champ() -> None:
    assert residu_de_la_fonction_de_courant(TEMOIN) == (0, 0)


def test_le_potentiel_du_temoin_n_est_pas_celui_de_c() -> None:
    """Controle de non-trivialite : les residus nuls ne viennent pas d'un zero."""
    assert (
        sp.simplify(
            TEMOIN.potentiel_des_vitesses() - enonce.CHAMP_C.potentiel_des_vitesses(),
        )
        != 0
    )


def test_la_loi_de_volume_verifie_l_equation_dont_elle_sort() -> None:
    """(1/V) dV/dt = div V : la definition, et non la forme close attendue.

    Le test existant compare la loi a `exp(t^2/2 - t)` ; il verifie donc
    l'integration. Celui-ci verifie qu'elle resout bien l'equation de
    depart -- controle equivalent en apparence, independant en pratique :
    il ne suppose rien de la forme du resultat.
    """
    for champ in (enonce.CHAMP_B, enonce.CHAMP_C):
        loi = champ.loi_de_volume()
        taux = sp.simplify(sp.diff(loi, T) / loi)
        assert sp.simplify(taux - champ.divergence()) == 0


def test_la_loi_de_volume_part_bien_de_un() -> None:
    """V(0)/V_0 = 1 : la condition initiale doit etre satisfaite."""
    for champ in (enonce.CHAMP_B, enonce.CHAMP_C):
        assert sp.simplify(champ.loi_de_volume().subs(T, 0) - 1) == 0
