"""Exercice cote 2025 (Charleroi), question 2 : une spire qui glisse sur un barreau.

Le champ du barreau decroit le long de l'axe ; la spire, en se
deplacant, embrasse un flux qui change, et c'est ce qui l'induit.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

T: Final = TEMPS
S: Final = sp.Symbol("S", positive=True)
PULSATION: Final = 10 * sp.pi
FEM_VOULUE: Final = sp.Rational(2, 1000) * sp.cos(PULSATION * T)
"""`2 cos(10 pi t)` mV, en volts."""
COURSE: Final = sp.Rational(1, 10)
"""La valeur maximale voulue pour `x_s`, en metres."""


def champ(position: sp.Expr) -> sp.Expr:
    """`H(x) = 20 - 2x`, en A/m."""
    return 20 - 2 * position


def induction(position: sp.Expr) -> sp.Expr:
    """`B = 1,1 + 0,05 (H - 20)`, en teslas : la droite donnee autour de 20 A/m."""
    return sp.Rational(11, 10) + sp.Rational(5, 100) * (champ(position) - 20)


def flux(position: sp.Expr) -> sp.Expr:
    """Le flux a travers la spire, uniforme sur la section : `B S`."""
    return induction(position) * S


def position() -> sp.Expr:
    """`x_s(t)` : ce que `-d(phi)/dt = E` impose, avec `x_s(0) = 0`."""
    inconnue = sp.Function("x")
    equation = sp.Eq(force_electromotrice(flux(inconnue(T))), FEM_VOULUE)
    solution = sp.dsolve(equation, inconnue(T), ics={inconnue(0): 0})
    return sp.simplify(solution.rhs)


def section() -> sp.Expr:
    """`S`, en m^2, pour que `x_s` culmine a dix centimetres."""
    amplitude = position().subs(T, sp.pi / (2 * PULSATION))
    (valeur,) = sp.solve(sp.Eq(amplitude, COURSE), S)
    return valeur


def fem_obtenue() -> sp.Expr:
    """La force electromotrice que le mouvement trouve induit reellement."""
    return force_electromotrice(flux(position()))


def plage_du_champ() -> tuple[sp.Expr, sp.Expr]:
    """Les valeurs extremes de `H` sur la course, en A/m."""
    return champ(COURSE), champ(-COURSE)


def vitesse_maximale() -> sp.Expr:
    """La plus grande vitesse de la spire, en m/s."""
    return sp.simplify(position().diff(T).subs(T, 0).subs(S, section()))
