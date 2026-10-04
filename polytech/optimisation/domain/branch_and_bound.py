"""Chapitre XIV.3 : la separation et l'evaluation d'un programme en variables entieres.

L'**evaluation** d'un noeud est la valeur optimale de sa relaxation
lineaire. La **separation** porte sur une variable `x_l` astreinte a etre
entiere et qui ne l'est pas dans la solution du probleme relaxe : deux
sous-noeuds, l'un avec `x_l <= [a_l0]`, l'autre avec `x_l >= [a_l0] + 1`.
La solution du parent ne verifie ni l'une ni l'autre, et aucun point
entier n'est perdu.

Ce module travaille sur les programmes ; `separation_sur_tableau` fait la
meme separation sur le tableau simplexe, ou l'algorithme dual
reoptimise en quelques pivots.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.separation_evaluation import (
    Cheminement,
    Procedure,
    explorer,
)
from polytech.optimisation.domain.sondage import Evaluation, Exploration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.entiers import ProgrammeEntier

type Point = tuple[sp.Expr, ...]


def evaluer_par_relaxation_lineaire(probleme: ProgrammeEntier) -> Evaluation[Point]:
    """`v_i = z_LP` ; la solution est admissible si elle est deja entiere."""
    solution = resoudre(probleme.relaxation_lineaire())
    if solution.issue is Issue.IMPOSSIBLE:
        return Evaluation(None, None, admissible=False)
    if solution.point is None:
        msg = "relaxation lineaire non bornee : aucune borne inferieure a propager"
        raise ValueError(msg)
    entier = probleme.est_entier(solution.point)
    return Evaluation(solution.valeur, solution.point, admissible=entier)


def separer_sur_une_variable(
    probleme: ProgrammeEntier,
    variable: int,
    valeur: sp.Expr,
) -> tuple[ProgrammeEntier, ProgrammeEntier]:
    """Les deux sous-noeuds : `x_l <= [a_l0]` et `x_l >= [a_l0] + 1`."""
    n = len(probleme.programme.objectif)
    ligne = tuple(int(j == variable) for j in range(n))
    plancher = sp.floor(valeur)
    bornes = (
        Contrainte(ligne, Sens.INFERIEUR, plancher),
        Contrainte(ligne, Sens.SUPERIEUR, plancher + 1),
    )
    gauche, droite = (
        replace(
            probleme,
            programme=replace(
                probleme.programme,
                contraintes=(*probleme.programme.contraintes, borne),
            ),
        )
        for borne in bornes
    )
    return gauche, droite


def separer_le_noeud(
    probleme: ProgrammeEntier,
    evaluation: Evaluation[Point],
) -> Sequence[ProgrammeEntier]:
    """Separe sur la premiere variable astreinte qui n'est pas entiere."""
    if evaluation.solution is None:
        msg = "noeud sans solution relaxee : il n'y a rien a separer"
        raise ValueError(msg)
    point = evaluation.solution
    variable = min(j for j in probleme.entieres if not point[j].is_integer)
    return separer_sur_une_variable(probleme, variable, point[variable])


def resoudre_en_variables_entieres(
    probleme: ProgrammeEntier,
    cheminement: Cheminement = Cheminement.PROFONDEUR_D_ABORD,
) -> Exploration[ProgrammeEntier, Point]:
    """L'optimum entier (ou mixte) par separation et evaluation.

    La procedure minimise. Une maximisation est exploree comme la
    minimisation de `-z` : les evaluations des noeuds sont alors celles de
    `-z`, et seule la valeur finale est rendue dans le sens d'origine.
    """
    procedure = Procedure(evaluer_par_relaxation_lineaire, separer_le_noeud)
    if probleme.programme.optimisation is Optimisation.MIN:
        return explorer(probleme, procedure, cheminement)
    oppose = replace(probleme, programme=probleme.programme.en_minimisation())
    exploration = explorer(oppose, procedure, cheminement)
    valeur = None if exploration.valeur is None else -exploration.valeur
    return replace(exploration, valeur=valeur)
