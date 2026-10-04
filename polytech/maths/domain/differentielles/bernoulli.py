"""L'equation de Bernoulli, `y' + P(x) y = Q(x) y^a`.

Avec `a` different de 0 et de 1 -- sinon elle est deja lineaire. On
divise par `y^a` et l'on pose `z = y^(1 - a)` : `z' = (1 - a) y^(-a) y'`,
d'ou une equation **lineaire** en `z`,

    z' + (1 - a) P(x) z = (1 - a) Q(x).

Deux ecueils (remarque 1.1 du syllabus) : la division par `y^a` elimine la
solution singuliere `y = 0` ; et il faut la solution **generale** en `z`
avant de revenir a `y`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.differentielles.lineaire import EquationLineaire

DEJA_LINEAIRES = (0, 1)


@dataclass(frozen=True, slots=True)
class EquationDeBernoulli:
    """`P`, `Q` et l'exposant `a` de `y' + P y = Q y^a`."""

    p: sp.Expr
    q: sp.Expr
    exposant: sp.Expr

    def __post_init__(self) -> None:
        """Refuse `a = 0` et `a = 1`."""
        if sp.sympify(self.exposant) in DEJA_LINEAIRES:
            msg = f"equation deja lineaire pour a = {self.exposant} : pas de Bernoulli"
            raise ValueError(msg)

    def equation_en_z(self) -> EquationLineaire:
        """`z' + (1 - a) P z = (1 - a) Q`, avec `z = y^(1 - a)`."""
        facteur = 1 - sp.sympify(self.exposant)
        return EquationLineaire(sp.simplify(facteur * self.p), facteur * self.q)

    def solution_generale(self) -> sp.Expr:
        """`y = z^(1 / (1 - a))`, `z` solution generale de l'equation lineaire."""
        z = self.equation_en_z().solution_generale()
        return sp.simplify(z ** (1 / (1 - sp.sympify(self.exposant))))

    def solution_singuliere(self) -> sp.Expr:
        """`y = 0`, que la division par `y^a` a ecartee."""
        return sp.Integer(0)
