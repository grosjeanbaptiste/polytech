"""Le modele predateur-proie de Lotka et Volterra.

Dans un ecosysteme ferme, `x(t)` proies et `y(t)` predateurs :

    x' =  a x - alpha x y,
    y' = -c y + gamma x y,

avec `a` le taux de croissance des proies sans predateur, `alpha` leur
mortalite due aux predateurs, `c` la mortalite des predateurs sans proie
et `gamma` leur croissance. Le systeme n'est pas lineaire ; ses points
critiques s'etudient par son linearise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import sympy as sp

from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE
from polytech.maths.domain.symboles import TEMPS


@dataclass(frozen=True, slots=True)
class PredateurProie:
    """Les quatre constantes positives du modele."""

    croissance: sp.Expr
    predation: sp.Expr
    mortalite: sp.Expr
    conversion: sp.Expr

    POPULATIONS: ClassVar[tuple[sp.Symbol, sp.Symbol]] = sp.symbols("x y", real=True)
    """Les proies `x` et les predateurs `y`."""

    def champ(self) -> tuple[sp.Expr, sp.Expr]:
        """`(x', y')`."""
        x, y = self.POPULATIONS
        return (
            self.croissance * x - self.predation * x * y,
            -self.mortalite * y + self.conversion * x * y,
        )

    def evolution_isolee(self) -> tuple[sp.Expr, sp.Expr]:
        """`(x(t), y(t))` quand l'autre population est absente.

        Proies seules : `x' - a x = 0` ; predateurs seuls : `y' + c y = 0`.
        Les deux sont lineaires du premier ordre, et la variable est le temps.
        """
        x, y = self.POPULATIONS
        proies, predateurs = self.champ()
        taux_x = sp.simplify(proies.subs(y, 0) / x)
        taux_y = sp.simplify(predateurs.subs(x, 0) / y)
        return (
            EquationLineaire(-taux_x, 0).solution_generale().subs(VARIABLE, TEMPS),
            EquationLineaire(-taux_y, 0).solution_generale().subs(VARIABLE, TEMPS),
        )

    def points_critiques(self) -> tuple[tuple[sp.Expr, sp.Expr], ...]:
        """Les couples ou les deux populations cessent de varier."""
        solutions = sp.solve(self.champ(), self.POPULATIONS, dict=True)
        x, y = self.POPULATIONS
        return tuple((solution[x], solution[y]) for solution in solutions)

    def linearise(self, point: tuple[object, object]) -> sp.Matrix:
        """La matrice jacobienne du champ, au point donne."""
        jacobienne = sp.Matrix(self.champ()).jacobian(self.POPULATIONS)
        return sp.simplify(
            jacobienne.subs(dict(zip(self.POPULATIONS, point, strict=True)))
        )
