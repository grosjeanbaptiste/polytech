"""Exercice cote 2017, question 3 : ouvrir une source sur une bobine.

Ferme, l'interrupteur met la bobine aux bornes de la source ; ouvert, il
laisse la bobine se decharger dans la resistance.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.phaseurs import valeur_instantanee
from polytech.electromagnetisme.domain.energie_et_forces import (
    energie_d_une_inductance,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

T = TEMPS
E, R, L, W = sp.symbols("E R L omega", positive=True)
THETA = sp.Symbol("theta", real=True)


def courant_avant() -> sp.Expr:
    """Le courant de la bobine en regime sinusoidal : `I = E / (j omega L)`."""
    phaseur = E * sp.exp(sp.I * THETA) / (sp.I * W * L)
    return valeur_instantanee(phaseur, W)


def courant_apres() -> sp.Expr:
    """`i(t) = i(0) e^(-R t / L)` : le courant d'une bobine ne saute pas."""
    depart = sp.simplify(courant_avant().subs(T, 0))
    return depart * sp.exp(-R * T / L)


def phases_d_extinction() -> tuple[sp.Expr, ...]:
    """Les phases, sur un tour, pour lesquelles le courant est nul a l'ouverture."""
    depart = courant_avant().subs(T, 0)
    racines = sp.solveset(depart, THETA, domain=sp.Interval.Ropen(0, 2 * sp.pi))
    return tuple(sorted(racines))


def energies() -> tuple[sp.Expr, sp.Expr]:
    """L'energie dissipee dans `R` apres l'ouverture, et celle de la bobine avant."""
    apres = courant_apres()
    dissipee = sp.integrate(R * apres**2, (T, 0, sp.oo))
    return sp.simplify(dissipee), energie_d_une_inductance(L, apres.subs(T, 0))
