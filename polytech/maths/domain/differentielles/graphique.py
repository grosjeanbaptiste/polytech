"""L'interpretation graphique de `y' = f(x, y)` (planches 25 a 28).

A tout point `M = (x0, y0)` l'equation associe une droite `D_M`, de pente
`f(x0, y0)` : c'est le **champ des tangentes**. Une **courbe integrale**
est une courbe tangente a `D_M` en chacun de ses points ; la solution
generale est l'ensemble des courbes integrales. Les **isoclines**
`Gamma_p : f(x, y) = p` sont les courbes le long desquelles la pente est
constante ; `Gamma_0` separe les solutions croissantes des decroissantes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE

if TYPE_CHECKING:
    from sympy.core.relational import Relational


def tangente(second_membre: sp.Expr, point: tuple[object, object]) -> sp.Eq:
    """`D_M : y - y0 = f(x0, y0) (x - x0)`."""
    x0, y0 = (sp.sympify(c) for c in point)
    pente = second_membre.subs({VARIABLE: x0, INCONNUE: y0})
    return sp.Eq(INCONNUE - y0, pente * (VARIABLE - x0))


def isocline(second_membre: sp.Expr, pente: object) -> sp.Eq:
    """`Gamma_p : f(x, y) = p`."""
    return sp.Eq(second_membre, sp.sympify(pente))


def regions_de_variation(second_membre: sp.Expr) -> tuple[Relational, Relational]:
    """`(f > 0, f < 0)` : ou les solutions croissent, ou elles decroissent."""
    return second_membre > 0, second_membre < 0
