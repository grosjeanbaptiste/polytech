"""L'equation de Clairaut, `y = x y' + phi(y')`.

Par la methode de la surface, avec `x = u` et `y' = v` pour parametres :
`(u + phi'(v)) dv = 0`. Deux issues :

* `dv = 0` : `y' = c`, la **famille de droites** `y = c x + phi(c)` ;
* `u = -phi'(v)` : la courbe `x = -phi'(c)`, `y = -c phi'(c) + phi(c)`,
  **enveloppe** de ces droites. Elle n'en fait pas partie : c'est une
  solution singuliere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import VARIABLE

PENTE: Final = sp.Symbol("z", real=True)
"""`z = y'`, la troisieme coordonnee de la methode de la surface."""


@dataclass(frozen=True, slots=True)
class EquationDeClairaut:
    """`phi`, ecrite en la pente `z`."""

    phi: sp.Expr

    def phi_en(self, pente: object) -> sp.Expr:
        """`phi(c)`."""
        return sp.sympify(self.phi).subs(PENTE, sp.sympify(pente))

    def droite(self, pente: object) -> sp.Expr:
        """`y = c x + phi(c)`, la solution de pente `c`."""
        return sp.sympify(pente) * VARIABLE + self.phi_en(pente)

    def solution_singuliere(self, parametre: object) -> tuple[sp.Expr, sp.Expr]:
        """`(x, y) = (-phi'(c), -c phi'(c) + phi(c))`."""
        c = sp.sympify(parametre)
        derivee = sp.diff(sp.sympify(self.phi), PENTE).subs(PENTE, c)
        return -derivee, sp.simplify(-c * derivee + self.phi_en(c))
