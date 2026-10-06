"""Mathematique, exercices 2 : une equation du premier ordre et sa solution.

Une solution se controle sans la recalculer : on la *reporte* dans
l'equation. Explicite, `y(x)` doit rendre `y' - f(x, y)` nul ; implicite,
`g(x, y) = K` doit rendre nul `dg/dx + f dg/dy`, la derivee de `g` le
long d'une courbe integrale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE

X: Final = VARIABLE
Y: Final = INCONNUE
K: Final = sp.Symbol("K", real=True)
"""La constante d'integration, dans la notation de la feuille."""


@dataclass(frozen=True, slots=True)
class Resolue:
    """`y' = pente(x, y)`, son type, les etapes et la solution."""

    pente: sp.Expr
    type: str
    etapes: tuple[sp.Basic, ...]
    solution: sp.Basic

    def residu(self) -> sp.Expr:
        """Ce qui reste de l'equation une fois la solution reportee : zero."""
        if isinstance(self.solution, sp.Equality):
            potentiel = self.solution.lhs
            le_long = potentiel.diff(X) + self.pente * potentiel.diff(Y)
            return sp.simplify(le_long)
        reportee = self.pente.subs(Y, self.solution)
        return sp.simplify(self.solution.diff(X) - reportee)
