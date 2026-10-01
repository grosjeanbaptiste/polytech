"""Resoudre un probleme de Cauchy du premier ordre par la transformee.

`y' + a y = f(t)`, `y(0) = y_0`. Ce que la methode apporte, et que le
corrige doit montrer, est l'**equation algebrique en p** : l'equation
differentielle devient une equation du premier degre, que l'on resout sans
integrer. C'est l'etape que `equation_en_image` expose.

La solution est ensuite verifiee en la reinjectant dans l'equation de
depart -- le seul controle qui ne depende d'aucune table d'inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE, TEMPS

INCONNUE_IMAGE: Final = sp.Symbol("Y", real=True)
"""`Y(p)`, l'image de l'inconnue, que le cours ecrit ainsi."""


@dataclass(frozen=True, slots=True)
class ProblemeDeCauchy:
    """`y' + a y = f(t)` avec `y(0)` donne."""

    coefficient: sp.Expr
    second_membre: sp.Expr
    valeur_initiale: sp.Expr

    @property
    def image(self) -> sp.Symbol:
        """Le symbole `Y(p)` de l'inconnue transformee."""
        return INCONNUE_IMAGE

    def equation_en_image(self) -> sp.Expr:
        """L'equation algebrique en `p`, egalee a zero.

        `p Y - y_0 + a Y - F(p) = 0`. C'est la ligne que le corrige
        montre : l'equation differentielle est devenue du premier degre.
        """
        derivee = Transformee.de_la_derivee(INCONNUE_IMAGE, self.valeur_initiale)
        second = Transformee.de(self.second_membre).image
        return sp.expand(derivee + self.coefficient * INCONNUE_IMAGE - second)

    def solution(self) -> sp.Expr:
        """La solution `y(t)` sur `t >= 0`, par inversion de l'image resolue.

        L'inversion rend un facteur `Heaviside(t)` que le cours n'ecrit
        jamais, et il a raison : la transformee de Laplace est
        **unilaterale**, elle ne connait que `t >= 0`, et sur ce domaine
        l'echelon vaut un. Le garder ferait trainer un `theta(t)` sur
        chaque ligne du corrige sans y rien signifier.

        Seul `Heaviside(t)` est reduit. Un echelon **retarde**,
        `Heaviside(t - a)`, porte lui une information -- l'instant ou la
        sollicitation commence -- et reste donc intact.
        """
        resolue = sp.solve(self.equation_en_image(), INCONNUE_IMAGE)[0]
        inversee = sp.inverse_laplace_transform(resolue, IMAGE, TEMPS, noconds=True)
        return sp.simplify(inversee.subs(sp.Heaviside(TEMPS), 1))

    def solution_verifiee(self) -> bool:
        """Vrai si la solution satisfait l'equation et la condition initiale."""
        solution = self.solution()
        reste = sp.diff(solution, TEMPS) + self.coefficient * solution
        satisfait = sp.simplify(reste - self.second_membre) == 0
        initiale = sp.simplify(solution.subs(TEMPS, 0) - self.valeur_initiale) == 0
        return bool(satisfait and initiale)
