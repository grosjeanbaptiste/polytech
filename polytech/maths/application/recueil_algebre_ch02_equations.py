"""Recueil d'algebre, chapitre II : equations matricielles et matrices particulieres.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 8
(sections II.5 et II.6).
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.matrices_particulieres import (
    commutant,
    parties_symetrique_et_antisymetrique,
)
from polytech.maths.domain.symboles import VARIABLE as X

M = sp.Matrix
GAUCHE_1: Final = M([[2, 3, 4], [5, 1, 2], [0, 3, 5]])
DROITE_1: Final = M([[2, 1], [3, 2], [4, 0]])
SECOND_1: Final = M([[0, 2], [1, 4], [2, 1]])
DONNEES_2: Final = (M([[3, -2], [0, 1]]), M([[-3, 1], [2, 0]]), M([[0, -1], [2, 1]]))
"""Les matrices `A`, `B` et `C` de l'exercice 2."""
A_3: Final = M([[3, 1, 0], [0, 3, 1], [0, 0, 3]])
A_5: Final = M([[0, 3], [3, 4]])
A_6: Final = M([[1, 2], [3, 4]])
THETA: Final = sp.Symbol("vartheta", real=True)
ROTATION: Final = M([[sp.cos(THETA), -sp.sin(THETA)], [sp.sin(THETA), sp.cos(THETA)]])
A_8: Final = M([[X, X / 2 - sp.Rational(1, 3)], [3, X - 1]])


def exercice_1() -> sp.Matrix:
    """`X = (G D - S) / 3`, de `-3X + G D = S`."""
    return (GAUCHE_1 * DROITE_1 - SECOND_1) / 3


def exercice_2() -> sp.Matrix:
    """La matrice `X` telle que `A (X - B) = X C`, inconnue par inconnue."""
    a, b, c = DONNEES_2
    inconnues = sp.symbols("x1:5")
    generale = M(2, 2, inconnues)
    (solution,) = sp.solve(
        list(a * (generale - b) - generale * c), inconnues, dict=True
    )
    return generale.subs(solution)


def exercice_3() -> sp.Matrix:
    """La matrice generale qui commute avec `A`."""
    return commutant(A_3)


def exercice_4() -> sp.Matrix:
    """Les triangulaires `(x y ; 0 z)` qui commutent avec toute matrice 2 x 2."""
    x, y, z = sp.symbols("x y z")
    triangulaire = M([[x, y], [0, z]])
    conditions: list[sp.Expr] = []
    for base in (M([[0, 1], [0, 0]]), M([[0, 0], [1, 0]])):
        conditions += list(triangulaire * base - base * triangulaire)
    (solution,) = sp.solve(conditions, [y, z], dict=True)
    return triangulaire.subs(solution)


def exercice_5() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Le terme hors diagonale de `X^t A X`, et les `x` qui l'annulent."""
    inconnue = M([[1, -X], [X, 1]])
    hors_diagonale = sp.expand((inconnue.T * A_5 * inconnue)[0, 1])
    return hors_diagonale, tuple(sp.solve(hors_diagonale, X))


def exercice_6() -> tuple[sp.Matrix, sp.Matrix]:
    """Les parties symetrique et antisymetrique de `A`."""
    return parties_symetrique_et_antisymetrique(A_6)


def exercice_7() -> sp.Matrix:
    """`R R^t`, simplifie : l'identite si la rotation est orthogonale."""
    return sp.simplify(ROTATION * ROTATION.T)


def exercice_8() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Le determinant, et les `x` qui l'annulent."""
    determinant = sp.expand(A_8.det())
    return determinant, tuple(sp.solve(determinant, X))
