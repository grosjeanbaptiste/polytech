"""Exercice I du jeu « algorithme simplexe » : une variable bornee des deux cotes.

`min 4x - 3y` sous trois contraintes, avec `-3 <= x <= 2` et `y >= 0`. La
variable `x` peut etre negative : on l'ecrit `x+ - x-`, ou bien on la
translate en `x' = x + 3`. Les tableaux des planches sont rejoues base par
base, l'artificielle etant penalisee par un `M` symbolique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import FormeStandard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.tableau import tableau_de_la_base

M: Final = sp.Symbol("M", positive=True)
INF, SUP, EGAL = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
PROGRAMME: Final = ProgrammeGeneral(
    Optimisation.MIN,
    (4, -3),
    (
        Contrainte((-1, 1), INF, 6),
        Contrainte((1, 1), EGAL, 4),
        Contrainte((-2, 4), SUP, 2),
        Contrainte((1, 0), INF, 2),
        Contrainte((1, 0), SUP, -3),
    ),
    libres=frozenset({0}),
)
TRANSLATE: Final = ProgrammeGeneral(
    Optimisation.MIN,
    (4, -3),
    (
        Contrainte((-1, 1), INF, 3),
        Contrainte((1, 1), EGAL, 7),
        Contrainte((2, -4), INF, 4),
        Contrainte((1, 0), INF, 5),
    ),
)
"""Le meme probleme en `x' = x + 3` ; l'objectif y vaut `-12 + 4x' - 3y`."""
DECALAGE: Final = 3
BASES: Final = ((4, 8, 9, 6, 7), (4, 8, 3, 6, 7), (4, 8, 3, 1, 7), (4, 5, 3, 1, 7))
"""Planches 6 a 9 ; les colonnes sont `x+ x- y x1e x2e x3e x4e x1a x2a`."""
BASES_FINALES: Final = ((4, 5, 3, 6, 7), (2, 5, 3, 6, 7))
BASES_TRANSLATEES: Final = ((3, 6, 4, 5), (2, 6, 4, 5), (2, 1, 4, 5))
"""Planche 14 ; les colonnes sont `x' y x1e x2e x3e xa`."""


@dataclass(frozen=True, slots=True)
class Etape:
    """Ce qu'un tableau dit : le point, la valeur, et s'il est admissible."""

    point: tuple[sp.Expr, sp.Expr]
    valeur: sp.Expr
    admissible: bool


def forme() -> FormeStandard:
    """La forme standard des planches, artificielles comprises."""
    matrice = sp.Matrix(
        [
            [-1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, -1, 1, 0, 0, 0, 0, 1, 0],
            [-2, 2, 4, 0, -1, 0, 0, 0, 1],
            [1, -1, 0, 0, 0, 1, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0, 1, 0, 0],
        ]
    )
    couts = sp.Matrix([[4, -4, -3, 0, 0, 0, 0, M, M]])
    return FormeStandard(couts, matrice, sp.Matrix([6, 4, 2, 2, 3]))


def parcours() -> tuple[Etape, ...]:
    """Les six tableaux des planches 6 a 11, dans l'ordre."""
    standard, etapes = forme(), []
    for base in (*BASES, *BASES_FINALES):
        tableau = tableau_de_la_base(standard, base)
        s = tableau.solution(standard.matrice.cols)
        point = (s[0] - s[1], s[2])
        etapes.append(Etape(point, tableau.valeur, bool(s[7] == 0 and s[8] == 0)))
    return tuple(etapes)


def parcours_translate() -> tuple[Etape, ...]:
    """Les trois tableaux de la planche 14, en `(x', y)`."""
    matrice = sp.Matrix(
        [
            [-1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1],
            [2, -4, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
        ]
    )
    standard = FormeStandard(
        sp.Matrix([[4, -3, 0, 0, 0, M]]), matrice, sp.Matrix([3, 7, 4, 5])
    )
    etapes = []
    for base in BASES_TRANSLATEES:
        tableau = tableau_de_la_base(standard, base)
        s = tableau.solution(standard.matrice.cols)
        valeur = tableau.valeur - 4 * DECALAGE
        etapes.append(Etape((s[0], s[1]), valeur, bool(s[5] == 0)))
    return tuple(etapes)


def optimum() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """La valeur et le point optimaux, par la bibliotheque."""
    return optimum_fini(PROGRAMME)


def optimum_translate() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Les memes, pour le probleme en `x'` ; la constante est retablie."""
    valeur, point = optimum_fini(TRANSLATE)
    return valeur - 4 * DECALAGE, point


def sommets_du_domaine() -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """Les deux extremites du segment admissible : `x` minimal, puis maximal."""
    extremites = []
    for sens in (Optimisation.MIN, Optimisation.MAX):
        selon_x = ProgrammeGeneral(
            sens, (1, 0), PROGRAMME.contraintes, PROGRAMME.libres
        )
        extremites.append(optimum_fini(selon_x)[1])
    return extremites[0], extremites[1]
