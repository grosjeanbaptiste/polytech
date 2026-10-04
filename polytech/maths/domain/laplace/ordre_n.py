"""Resoudre une EDO lineaire a coefficients constants par la transformee de Laplace.

L'idee des planches 3 et 15-16 : **EDO -> equation algebrique**. Chaque
derivee devient une multiplication par `p`, conditions initiales
comprises ; il reste a resoudre une equation du premier degre en `X(p)`,
puis a revenir a `x(t)` par la transformee inverse.

    x'' + 3x' + 2x = 1,  x(0) = x'(0) = 0
    (p^2 + 3p + 2) X(p) = 1/p
    X(p) = 1 / (p (p^2 + 3p + 2)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.laplace.proprietes import de_la_derivee_n
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE, TEMPS

INCONNUE: Final = sp.Symbol("X")
"""`X(p)`, l'image de l'inconnue."""


@dataclass(frozen=True, slots=True)
class ProblemeLineaire:
    """`a_n x^(n) + ... + a_0 x = f(t)`, et `x(0+)`, `x'(0+)`, ..."""

    coefficients: tuple[object, ...]
    second_membre: sp.Expr
    initiales: tuple[object, ...]

    def __post_init__(self) -> None:
        """Refuse un nombre de conditions initiales different de l'ordre."""
        ordre = len(self.coefficients) - 1
        if len(self.initiales) != ordre:
            msg = (
                f"{len(self.initiales)} conditions initiales pour une equation "
                f"d'ordre {ordre}"
            )
            raise ValueError(msg)

    @property
    def inconnue(self) -> sp.Symbol:
        """Le symbole `X(p)`."""
        return INCONNUE

    def equation_en_image(self) -> sp.Expr:
        """L'equation algebrique en `X(p)`, egalee a zero."""
        ordre = len(self.coefficients) - 1
        membre = sum(
            (
                sp.sympify(a) * de_la_derivee_n(INCONNUE, self.initiales[: ordre - k])
                for k, a in enumerate(self.coefficients)
            ),
            sp.Integer(0),
        )
        return sp.expand(membre - Transformee.de(self.second_membre).image)

    def image(self) -> sp.Expr:
        """`X(p)`, solution de l'equation algebrique."""
        return sp.simplify(sp.solve(self.equation_en_image(), INCONNUE)[0])

    def solution(self) -> sp.Expr:
        """`x(t)` pour `t >= 0`, par la transformee inverse."""
        inverse = sp.inverse_laplace_transform(self.image(), IMAGE, TEMPS, noconds=True)
        return sp.simplify(inverse.subs(sp.Heaviside(TEMPS), 1))

    def est_solution(self, candidate: sp.Expr) -> bool:
        """Vrai si la fonction satisfait l'equation et chaque condition initiale."""
        ordre = len(self.coefficients) - 1
        membre = sum(
            (
                sp.sympify(a) * sp.diff(candidate, TEMPS, ordre - k)
                for k, a in enumerate(self.coefficients)
            ),
            sp.Integer(0),
        )
        initiales = all(
            sp.simplify(sp.diff(candidate, TEMPS, k).subs(TEMPS, 0) - sp.sympify(v))
            == 0
            for k, v in enumerate(self.initiales)
        )
        return bool(sp.simplify(membre - self.second_membre) == 0 and initiales)

    def solution_verifiee(self) -> bool:
        """Vrai si la solution trouvee passe ce controle, independant de l'inversion."""
        return self.est_solution(self.solution())
