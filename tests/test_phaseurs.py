"""Du domaine de Laplace au regime sinusoidal etabli.

Le resolveur operationnel rend `X(p) = H(p) E` pour une source constante
`E`. Le regime sinusoidal en est la restriction a `p = j omega` : la
source devient le phaseur efficace, et la reponse le sien. C'est la
meme algebre, et la recrire a part donnerait deux resolveurs a garder
d'accord.

Les oracles sont ceux de tout cours d'electricite : le diviseur `R`-`C`,
la puissance d'une resistance, la valeur a l'instant zero d'un cosinus.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.phaseurs import (
    en_regime_sinusoidal,
    puissance_complexe,
    valeur_instantanee,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE

R, C, E, THETA = sp.symbols("R C E theta", positive=True)


def test_le_diviseur_rc_devient_son_phaseur() -> None:
    """`U_C = E / (1 + j omega R C)` : le diviseur que tout cours donne."""
    transformee = E / (1 + VARIABLE * R * C)
    phaseur = en_regime_sinusoidal(transformee, PULSATION)
    assert sp.simplify(phaseur - E / (1 + sp.I * PULSATION * R * C)) == 0


def test_une_pulsation_negative_est_refusee() -> None:
    """Une pulsation est positive : un signe moins cacherait une erreur."""
    with pytest.raises(ValueError, match="pulsation"):
        en_regime_sinusoidal(E / VARIABLE, sp.Integer(-1))


def test_une_pulsation_nulle_est_refusee() -> None:
    """A pulsation nulle une capacite est un circuit ouvert : autre chapitre."""
    with pytest.raises(ValueError, match="pulsation"):
        en_regime_sinusoidal(E / VARIABLE, sp.Integer(0))


def test_une_resistance_n_absorbe_que_de_l_actif() -> None:
    """`S = U I* = R |I|^2` : reel, sans reactif."""
    courant = sp.Integer(3) - 4 * sp.I
    assert puissance_complexe(R * courant, courant) == 25 * R


def test_une_bobine_n_absorbe_que_du_reactif() -> None:
    """`S = j omega L |I|^2` : la convention recepteur donne `Q > 0`."""
    inductance = sp.Symbol("L", positive=True)
    courant = sp.Integer(2)
    tension = sp.I * PULSATION * inductance * courant
    assert puissance_complexe(tension, courant) == 4 * sp.I * PULSATION * inductance


def test_la_puissance_conjugue_le_courant_et_non_la_tension() -> None:
    """`U I*` et non `U* I` : les deux ont des parties imaginaires opposees."""
    assert puissance_complexe(sp.I, sp.Integer(1)) == sp.I


def test_le_phaseur_d_un_cosinus_rend_son_cosinus() -> None:
    """`E e^{j theta}` efficace donne `E racine 2 cos(omega t + theta)`."""
    phaseur = E * sp.exp(sp.I * THETA)
    attendu = E * sp.sqrt(2) * sp.cos(PULSATION * TEMPS + THETA)
    assert sp.simplify(valeur_instantanee(phaseur, PULSATION) - attendu) == 0
