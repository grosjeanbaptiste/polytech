"""Optimisation libre pour `f : R -> R`.

Syllabus de calcul différentiel et intégral 1, section 5.4. Un extremum
local d'une fonction dérivable est un point critique, `f'(a) = 0`
(condition nécessaire). La dérivée seconde le caractérise quand elle ne
s'annule pas : `f''(a) > 0` donne un minimum, `f''(a) < 0` un maximum ; si
`f''(a) = 0` elle ne tranche pas, et le point est rendu `INDETERMINE` plutôt
que deviné.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X


class Nature(Enum):
    """Ce que la dérivée seconde dit d'un point critique."""

    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    INDETERMINE = "indetermine"


def _nature(seconde: sp.Expr) -> Nature:
    if seconde > 0:
        return Nature.MINIMUM
    if seconde < 0:
        return Nature.MAXIMUM
    return Nature.INDETERMINE


def extrema_libres(expression: sp.Expr) -> tuple[tuple[sp.Expr, Nature], ...]:
    """Les points critiques réels de `f`, par abscisse croissante, avec leur nature."""
    premiere = sp.diff(expression, X)
    seconde = sp.diff(premiere, X)
    critiques = sorted(sp.solveset(premiere, X, sp.S.Reals), key=float)
    return tuple((p, _nature(seconde.subs(X, p))) for p in critiques)
