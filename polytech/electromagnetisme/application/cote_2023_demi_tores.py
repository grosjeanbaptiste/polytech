"""Exercice cote 2023 (Charleroi), question 2 : un demi-tore qui s'eloigne.

Deux hypotheses de l'enonce : pas de dispersion (le flux est le meme
partout), et les entrefers dominent (le fer ne consomme aucun
ampere-tour). L'entrefer croit comme `l_e + v t`, et il y en a deux.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    flux_a_entrefers_dominants,
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

T: Final = TEMPS
MU0: Final = PERMEABILITE
N, COURANT, S = sp.symbols("N I S", positive=True)
ENTREFER, VITESSE = sp.symbols("l_e v", positive=True)


def _air() -> sp.Expr:
    """La longueur d'air traversee a l'instant `t` : deux entrefers."""
    return 2 * (ENTREFER + VITESSE * T)


def flux() -> sp.Expr:
    """`Phi(t)`, le flux commun aux deux demi-tores."""
    return flux_a_entrefers_dominants(N, COURANT, S, _air())


def fem() -> sp.Expr:
    """La force electromotrice de l'enroulement de droite : `-N dPhi/dt`."""
    return force_electromotrice(N * flux())


def inductance_mutuelle() -> sp.Expr:
    """`M = N Phi / I` : le flux que l'enroulement de droite embrasse, par ampere."""
    return sp.simplify(N * flux() / COURANT)


def aire_sous_la_fem() -> sp.Expr:
    """`int_0^oo E dt`, en volts-secondes."""
    return sp.simplify(sp.integrate(fem(), (T, 0, sp.oo)))
