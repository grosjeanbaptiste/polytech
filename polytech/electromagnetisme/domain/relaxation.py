"""La relaxation de la charge volumique d'un conducteur.

Le syllabus d'electricite generale (14.20). Dans un conducteur homogene
et au repos, la continuite et la loi d'Ohm donnent
`tau d(eta)/dt + eta = 0` (14.20.2), avec le temps de relaxation
`tau = epsilon rho` (14.20.1). La charge volumique s'eteint donc en
`eta_0 exp(-t / tau)` (14.20.3) : apres quelques `tau`, toute la charge
est a la surface, comme en electrostatique.

Pour le cuivre, `tau = 0,15 . 10^-18 s`. Le syllabus note que la
resistivite en courant continu ne vaut sans doute plus a cette echelle,
et que la duree reelle est plus longue, mais reste tres petite.
"""

from __future__ import annotations

import math
from typing import Any

import sympy as sp


def temps_de_relaxation(permittivite: float, resistivite: float) -> float:
    """`tau = epsilon rho` (14.20.1), en secondes."""
    if permittivite <= 0 or resistivite <= 0:
        msg = (
            f"permittivite ou resistivite nulle ou negative : "
            f"{permittivite} F/m, {resistivite} ohm.m"
        )
        raise ValueError(msg)
    return permittivite * resistivite


def charge_volumique_residuelle(initiale: Any, temps: Any, relaxation: Any) -> sp.Expr:  # noqa: ANN401 -- reels ou symboles
    """`eta = eta_0 exp(-t / tau)` (14.20.3)."""
    return initiale * sp.exp(-temps / relaxation)


def duree_de_relaxation(fraction: float, relaxation: float) -> float:
    """`tau ln(1 / fraction)` : le temps au bout duquel il reste cette fraction."""
    if not 0 < fraction <= 1:
        msg = f"fraction hors de ]0 ; 1] : {fraction}"
        raise ValueError(msg)
    return relaxation * math.log(1 / fraction)
