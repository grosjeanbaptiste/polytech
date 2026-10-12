"""Examen de janvier 2025 : une integrale, un systeme, une equation.

Source : `Mathematique/EXAMENS/Exam_2025_Janvier_Corrige.pdf`, enonce et
corrige officiel dans le meme document. Le corrige de ce depot le complete :
chaque valeur est recalculee ici, et confrontee a l'officielle par les tests.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.discussion_par_le_rang import discuter_par_le_rang
from polytech.maths.domain.algebre.solutions import solution_generale
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.integration.definie import IntegraleDefinie
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X

PARAMETRE: Final = sp.Symbol("a", real=True)
_A = PARAMETRE
SYSTEME: Final = SystemeLineaire(
    sp.Matrix([[_A, 1 - _A, 1 - _A], [_A, 1 + _A, 1 + _A], [1, 1, 1]]),
    sp.Matrix([_A**2, _A - _A**2, 1 - _A]),
)
PROBLEME: Final = ProblemeLineaire((1, -5, -14), -14 * T**2 - 10 * T - 26, (0, 13))


def integrale() -> tuple[tuple[sp.Expr, ...], sp.Expr, sp.Expr]:
    """Les fractions simples, la primitive, et la valeur entre 2 et 3."""
    integrande = 4 * X**2 / (X**4 - 1)
    fraction = FractionRationnelle(4 * X**2, X**4 - 1)
    definie = IntegraleDefinie(integrande, sp.Integer(2), sp.Integer(3))
    return fraction.termes(), fraction.primitive(), definie.valeur()


def systeme() -> dict[sp.Expr, tuple[sp.Expr, ...]]:
    """Les valeurs du parametre qui rendent le systeme compatible, et leurs droites."""
    discussion = discuter_par_le_rang(SYSTEME, PARAMETRE)
    return {
        valeur: solution_generale(discussion.en(valeur))
        for valeur in discussion.particulieres
    }


def equation() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """L'image de la solution, ses fractions simples, et la solution."""
    image = PROBLEME.image()
    return image, sp.apart(image, P), PROBLEME.solution()
