"""Mathematique, exercices 2, I.1 : les equations lineaires 5 a 7, et le registre.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, page 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.application.equations_a_exs_02 import (
    constantes_de_la_1,
    equation_1,
    equation_2,
    equation_3,
    equation_4,
    second_chemin_de_la_2,
)
from polytech.maths.application.resolue_exs_02 import K, Resolue, X, Y
from polytech.maths.domain.differentielles.homogenes import Z
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.symboles import CONSTANTE

if TYPE_CHECKING:
    from collections.abc import Callable

LINEAIRE = "lineaire du premier ordre"


def _lineaire(p: sp.Expr, q: sp.Expr) -> Resolue:
    """`y' + P y = Q` : la solution homogene, `K(x)`, la solution particuliere."""
    equation = EquationLineaire(p, q)
    homogene = equation.solution_homogene().subs(CONSTANTE, K)
    etapes = (
        homogene,
        equation.constante_variable(),
        equation.solution_particuliere(),
    )
    solution = equation.solution_generale().subs(CONSTANTE, K)
    return Resolue(q - p * Y, LINEAIRE, etapes, solution)


def equation_5() -> Resolue:
    """`y' + y cos x = cos x sin x`."""
    return _lineaire(sp.cos(X), sp.cos(X) * sp.sin(X))


def equation_6() -> Resolue:
    """`y' + y tan x = cos x`."""
    return _lineaire(sp.tan(X), sp.cos(X))


def equation_7() -> Resolue:
    """`x (x - 1) y' + (1 - 2x) y + x^2 = 0`, mise sous forme normale."""
    return _lineaire((1 - 2 * X) / (X * (X - 1)), -X / (X - 1))


EQUATIONS: Final[dict[int, Callable[[], Resolue]]] = {
    1: equation_1,
    2: equation_2,
    3: equation_3,
    4: equation_4,
    5: equation_5,
    6: equation_6,
    7: equation_7,
}

__all__ = [
    "EQUATIONS",
    "Z",
    "constantes_de_la_1",
    "equation_1",
    "equation_2",
    "equation_3",
    "equation_4",
    "equation_5",
    "equation_6",
    "equation_7",
    "second_chemin_de_la_2",
]
