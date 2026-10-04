"""Les equations lineaires du premier ordre, `y' + P(x) y = Q(x)` (planches 38-40).

Lineaire en `y` et `y'`, pas necessairement en `x`. On pose `y = u v` et
l'on choisit `v` solution de l'equation homogene : `v = K e^(-int P)`.
Il reste `u' v = Q`, d'ou la formule (1) :

    y(x) = e^(-int P dx) [ C + int Q e^(int P dx) dx ].

C'est la methode de la **variation des constantes** : la solution
generale de l'equation homogene (SGEH), `K e^(-int P)`, puis une solution
particuliere de l'equation non homogene (SPENH) cherchee sous la meme
forme avec `K` fonction de `x`, `K(x) = int Q e^(int P)`. Leur somme est
la solution generale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import VARIABLE
from polytech.maths.domain.symboles import CONSTANTE
from polytech.raisonnement.domain.etapes import Etape, Raisonnement

SOURCE: Final = (
    "Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf, planches 38-40"
)


@dataclass(frozen=True, slots=True)
class EquationLineaire:
    """`y' + P(x) y = Q(x)`."""

    p: sp.Expr
    q: sp.Expr

    def _noyau(self) -> sp.Expr:
        """`e^(-int P dx)`, la solution de l'equation homogene a une constante pres."""
        return sp.simplify(sp.exp(-sp.integrate(self.p, VARIABLE)))

    def solution_homogene(self) -> sp.Expr:
        """SGEH : `C e^(-int P dx)`."""
        return CONSTANTE * self._noyau()

    def constante_variable(self) -> sp.Expr:
        """`K(x) = int Q e^(int P dx) dx`."""
        return sp.simplify(sp.integrate(self.q / self._noyau(), VARIABLE))

    def solution_particuliere(self) -> sp.Expr:
        """SPENH : `K(x) e^(-int P dx)`."""
        return sp.simplify(self.constante_variable() * self._noyau())

    def solution_generale(self) -> sp.Expr:
        """La formule (1) : la somme de la SGEH et de la SPENH."""
        return self.solution_homogene() + self.solution_particuliere()

    def expliquer(self) -> Raisonnement:
        """La variation des constantes, etape par etape."""
        etapes = (
            Etape(
                f"y_SGEH = {self.solution_homogene()}",
                "equation homogene y' + P y = 0, a variables separees",
                SOURCE,
            ),
            Etape(
                f"K(x) = {self.constante_variable()}",
                "on cherche y = K(x) e^(-int P) : il reste K' e^(-int P) = Q",
                SOURCE,
            ),
            Etape(
                f"y_SPENH = {self.solution_particuliere()}",
                "c'est K(x) e^(-int P)",
                SOURCE,
            ),
        )
        conclusion = f"y = {self.solution_generale()}"
        return Raisonnement("Que vaut la solution generale ?", etapes, conclusion)
