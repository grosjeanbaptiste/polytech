"""Exercice I, planche 16 : ce que coute chaque changement de variable.

Un second probleme, `min 10x - 2y`, est ecrit de deux facons : avec
`x = x+ - x-`, puis avec `x = x' - 3`. La planche compte, pour chacune,
les variables, les ecarts, les artificielles et les contraintes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

INF: Final = Sens.INFERIEUR
DECALAGE: Final = 3
DECOMPOSE: Final = ProgrammeGeneral(
    Optimisation.MIN,
    (10, -2),
    (
        Contrainte((-2, 1), INF, 5),
        Contrainte((1, 2), INF, 3),
        Contrainte((-4, 3), INF, 11),
        Contrainte((1, 0), INF, 2),
        Contrainte((-1, 0), INF, 3),
    ),
    libres=frozenset({0}),
)
"""Avec `x = x+ - x-` : les deux bornes de `x` sont deux contraintes."""
TRANSLATE: Final = ProgrammeGeneral(
    Optimisation.MIN,
    (10, -2),
    (
        Contrainte((-2, 1), INF, -1),
        Contrainte((1, 2), INF, 6),
        Contrainte((-4, 3), INF, -1),
        Contrainte((1, 0), INF, 5),
    ),
)
"""Avec `x = x' - 3` : la borne inferieure devient `x' >= 0`."""


def demande_une_artificielle(contrainte: Contrainte) -> bool:
    """Vrai si, le second membre rendu positif, l'ecart ne peut pas etre en base."""
    if contrainte.sens is Sens.EGAL:
        return True
    retournee = contrainte.second_membre < 0
    return bool((contrainte.sens is Sens.SUPERIEUR) != retournee)


def decompte(*, translate: bool) -> tuple[int, int, int, int]:
    """Variables, ecarts, artificielles et contraintes de la formulation."""
    programme = TRANSLATE if translate else DECOMPOSE
    variables = len(programme.objectif) + len(programme.libres)
    ecarts = sum(1 for c in programme.contraintes if c.sens is not Sens.EGAL)
    artificielles = sum(1 for c in programme.contraintes if demande_une_artificielle(c))
    return variables, ecarts, artificielles, len(programme.contraintes)


def optimum(*, translate: bool) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """La valeur et le point `(x, y)` optimaux, quelle que soit la formulation."""
    valeur, point = optimum_fini(TRANSLATE if translate else DECOMPOSE)
    if not translate:
        return valeur, point
    x_prime, y = point
    return valeur - 10 * DECALAGE, (x_prime - DECALAGE, y)
