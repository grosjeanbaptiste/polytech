"""Resoudre un programme lineaire sous sa forme generale, de bout en bout.

Le cours enseigne chaque maillon a part ; ce module les enchaine, dans
l'ordre ou il les enseigne :

1. ramener les variables de signe quelconque a des variables non
   negatives (I.1) ;
2. ramener une maximisation a une minimisation (I.1) ;
3. ecrire la forme standard, variables d'ecart comprises (I.1) ;
4. chercher une base admissible par les variables artificielles (II.5),
   puis l'optimum par la phase II (II.4).

Et il fait le chemin inverse : la solution est rendue dans les variables
d'origine -- une variable libre retrouve son signe, `x_j = x_j+ - x_j-`
-- et la valeur dans le sens d'origine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.phase_un import methode_des_variables_artificielles

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.formulation import ProgrammeGeneral


@dataclass(frozen=True, slots=True)
class Solution:
    """L'issue, et -- si elle est optimale -- la valeur et le point d'origine."""

    issue: Issue
    valeur: sp.Expr | None = None
    point: tuple[sp.Expr, ...] | None = None


def _point_d_origine(
    programme: ProgrammeGeneral,
    standard: sp.Matrix,
) -> tuple[sp.Expr, ...]:
    """Les variables du programme, les libres recomposees en `x+ - x-`."""
    variables = len(programme.objectif)
    negatives = {j: variables + rang for rang, j in enumerate(sorted(programme.libres))}
    return tuple(
        standard[j, 0] - (standard[negatives[j], 0] if j in negatives else 0)
        for j in range(variables)
    )


def resoudre(programme: ProgrammeGeneral) -> Solution:
    """L'optimum du programme, dans ses variables et son sens d'origine."""
    forme = forme_standard(programme)
    resolution = methode_des_variables_artificielles(forme, Optimisation.MIN)
    if resolution.issue is not Issue.OPTIMALE:
        return Solution(resolution.issue)
    final = resolution.tableaux[-1]
    point = _point_d_origine(programme, final.solution(forme.matrice.cols))
    return Solution(Issue.OPTIMALE, programme.valeur(point), point)
