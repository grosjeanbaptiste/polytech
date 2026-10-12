"""Examen de janvier 2021, question 2 : deux systemes a discuter.

Source : `Mathematique/EXAMENS/Exam_2021_Janvier_Enonce.pdf`, systemes (4)
et (5). Le systeme (6) est celui de janvier 2025 et se lit dans son corrige.
Le (4) a un determinant nul pour tout parametre : il se discute par le rang.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.discussion_par_le_rang import (
    DiscussionParLeRang,
    discuter_par_le_rang,
)
from polytech.maths.domain.algebre.solutions import solution_generale
from polytech.maths.domain.algebre.systemes import SystemeLineaire

PARAMETRE: Final = sp.Symbol("a", real=True)
_A = PARAMETRE


def _systeme(lignes: list[list[sp.Expr]]) -> SystemeLineaire:
    complete = sp.Matrix(lignes)
    return SystemeLineaire(complete[:, :-1], complete[:, -1])


SYSTEME_4: Final = _systeme(
    [
        [_A, _A + 1, _A - 1, 2 * _A + 3],
        [1, 1 + _A, 1 - _A, 4 * _A + 1],
        [_A + 1, 2 * (_A + 1), 0, 6 * _A + 4],
    ]
)
SYSTEME_5: Final = _systeme(
    [[_A, _A**2, _A**3, 0], [1, 1, 1, 0], [_A**3, _A**2, _A, 0]]
)


def discussion_4() -> DiscussionParLeRang:
    """La nature du systeme (4) pour toute valeur du parametre."""
    return discuter_par_le_rang(SYSTEME_4, PARAMETRE)


def solution_4_generique() -> tuple[sp.Expr, ...]:
    """La droite de solutions, hors de `a = -1` et de `a = 1`."""
    return tuple(sp.simplify(c) for c in solution_generale(SYSTEME_4))


def solution_4_en(valeur: sp.Expr) -> tuple[sp.Expr, ...]:
    """Les solutions du systeme (4) en cette valeur du parametre."""
    return solution_generale(discussion_4().en(valeur))


def determinant_5() -> sp.Expr:
    """Le determinant du systeme (5), factorise."""
    return sp.factor(SYSTEME_5.matrice.det())


def solution_5_en(valeur: sp.Expr) -> tuple[sp.Expr, ...]:
    """Les solutions du systeme (5) en cette valeur du parametre."""
    return solution_generale(discuter_par_le_rang(SYSTEME_5, PARAMETRE).en(valeur))
