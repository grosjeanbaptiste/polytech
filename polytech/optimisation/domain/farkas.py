"""Le lemme de Farkas (1902), tel que le chapitre IV le deduit de la dualite.

Le systeme `T x = d`, `x >= 0` n'a pas de solution **si et seulement si**
le systeme `u T <= 0`, `u d > 0` en a une. Des deux, un et un seul est
resoluble : c'est une **alternative**.

La demonstration du cours (pages 96-97) est constructive, et ce module la
suit : avec `c = 0`, le primal `min 0 x` a l'optimum 0 ou est impossible ;
dans le second cas le dual `max u d`, `u T <= 0`, qui admet `u = 0`, n'est
pas borne -- il existe donc `u` avec `u d > 0`. On le normalise par
`u d = 1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre

if TYPE_CHECKING:
    import sympy as sp


class Alternative(Enum):
    """Lequel des deux systemes a une solution."""

    SOLUTION_NON_NEGATIVE = "T x = d a une solution x >= 0"
    CERTIFICAT = "il existe u avec u T <= 0 et u d > 0"


@dataclass(frozen=True, slots=True)
class Temoin:
    """Le cote de l'alternative, et le vecteur qui le prouve (`x` ou `u`)."""

    cote: Alternative
    temoin: tuple[sp.Expr, ...]


def _admissible(contraintes: tuple[Contrainte, ...], libres: bool) -> ProgrammeGeneral:  # noqa: FBT001 -- interne : `x >= 0` d'un cote, `u` libre de l'autre
    variables = len(contraintes[0].coefficients)
    return ProgrammeGeneral(
        Optimisation.MIN,
        (0,) * variables,
        contraintes,
        frozenset(range(variables)) if libres else frozenset(),
    )


def alternative_de_farkas(matrice: sp.Matrix, second: sp.Matrix) -> Temoin:
    """Une solution `x >= 0` de `T x = d`, ou a defaut le certificat `u`."""
    lignes = range(matrice.rows)
    systeme = tuple(
        Contrainte(tuple(matrice[i, :]), Sens.EGAL, second[i, 0]) for i in lignes
    )
    primal = resoudre(_admissible(systeme, libres=False))
    if primal.issue is Issue.OPTIMALE and primal.point is not None:
        return Temoin(Alternative.SOLUTION_NON_NEGATIVE, primal.point)
    colonnes = tuple(
        Contrainte(tuple(matrice[:, j]), Sens.INFERIEUR, 0) for j in range(matrice.cols)
    )
    normalisation = Contrainte(tuple(second[:, 0]), Sens.EGAL, 1)
    dual = resoudre(_admissible((*colonnes, normalisation), libres=True))
    if dual.point is None:
        msg = "ni solution ni certificat : contraire au lemme de Farkas"
        raise RuntimeError(msg)
    return Temoin(Alternative.CERTIFICAT, dual.point)
