r"""Rendu LaTeX dans la notation du cours.

sympy ecrit un champ abstrait ``u(x, y, t)`` avec sa liste d'arguments, et
une derivee sous la forme ``\frac{\partial}{\partial x} u(x,y,t)``. Le
cours, lui, ecrit ``u`` et ``\frac{\partial u}{\partial x}``. Cet
imprimeur retablit cette notation sans toucher au calcul.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sympy.printing.latex import LatexPrinter

if TYPE_CHECKING:
    import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y

_ARGUMENTS_DU_CHAMP = (X, Y, T)


def _est_champ(expression: sp.Expr) -> bool:
    """Vrai pour une fonction des seules coordonnees et du temps."""
    return bool(expression.args == _ARGUMENTS_DU_CHAMP)


class ImprimeurDuCours(LatexPrinter):
    """Imprimeur LaTeX suivant les conventions d'ecriture du cours."""

    # `exp` est impose par la signature de sympy : son type y est libre.
    # N802/ANN401 : la signature est imposee par l'API de sympy, qui
    # appelle `_print_<Classe>` avec un `exp` de type libre.
    def _print_Function(self, expr: sp.Expr, exp: Any = None) -> str:  # noqa: ANN401, N802
        if _est_champ(expr) and exp is None:
            return str(expr.func.__name__)
        return str(super()._print_Function(expr, exp))

    # N802 : nom impose par le repartiteur de sympy (`_print_Mul`).
    def _print_Mul(self, expr: sp.Expr) -> str:  # noqa: N802
        """Ecrit ``1/2 (a + b)`` en facteur, et non distribue sur la somme."""
        coefficient, reste = expr.as_coeff_Mul()
        if coefficient.is_Rational and coefficient != 1 and reste.is_Add:
            return rf"{self._print(coefficient)}\left({self._print(reste)}\right)"
        return str(super()._print_Mul(expr))

    # N802 : nom impose par le repartiteur de sympy (`_print_Derivative`).
    def _print_Derivative(self, expr: sp.Derivative) -> str:  # noqa: N802
        if len(expr.variables) == 1:
            base = self._print(expr.expr)
            variable = self._print(expr.variables[0])
            return rf"\frac{{\partial {base}}}{{\partial {variable}}}"
        return str(super()._print_Derivative(expr))


def latex_du_cours(expression: sp.Expr) -> str:
    """Rend une expression avec la notation du cours."""
    return str(ImprimeurDuCours().doprint(expression))
