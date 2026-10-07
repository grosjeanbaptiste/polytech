"""Exercices V et VI du jeu « algorithme simplexe » : « Branch and Bound ».

Deux maximisations en variables entieres. Les planches minimisent l'oppose
de l'objectif : l'evaluation d'un sommet de l'arborescence est donc
l'oppose du maximum de sa relaxation lineaire.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier, points_entiers
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.penalites import penalites
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.tableau import tableau_de_la_base

if TYPE_CHECKING:
    from polytech.optimisation.domain.algorithme import Issue

INF: Final = Sens.INFERIEUR
CINQ: Final = ProgrammeGeneral(
    Optimisation.MAX,
    (3, 5),
    (Contrainte((1, 1), INF, 6), Contrainte((-2, 4), INF, 3)),
)
SIX: Final = ProgrammeGeneral(
    Optimisation.MAX,
    (6, 2, 9),
    (Contrainte((1, 1, 3), INF, 19), Contrainte((3, 1, 1), INF, 15)),
)
Borne = tuple[int, Sens, int]
"""Une contrainte de separation : la variable (de 1), le sens, la valeur."""


@dataclass(frozen=True, slots=True)
class Noeud:
    """Un sommet de l'arborescence : l'issue, le point, l'evaluation a minimiser."""

    issue: Issue
    point: tuple[sp.Expr, ...] | None
    evaluation: sp.Expr | None


def _borne(probleme: ProgrammeGeneral, borne: Borne) -> Contrainte:
    variable, sens, valeur = borne
    if not 1 <= variable <= len(probleme.objectif):
        msg = f"variable hors du probleme : x{variable}"
        raise ValueError(msg)
    coefficients = [0] * len(probleme.objectif)
    coefficients[variable - 1] = 1
    return Contrainte(tuple(coefficients), sens, valeur)


def noeud(probleme: ProgrammeGeneral, bornes: tuple[Borne, ...] = ()) -> Noeud:
    """La relaxation lineaire du probleme, ces contraintes de separation ajoutees."""
    ajoutees = tuple(_borne(probleme, b) for b in bornes)
    restreint = replace(probleme, contraintes=(*probleme.contraintes, *ajoutees))
    solution = resoudre(restreint)
    if solution.valeur is None or solution.point is None:
        return Noeud(solution.issue, None, None)
    return Noeud(solution.issue, solution.point, -solution.valeur)


def penalites_a_la_racine(
    probleme: ProgrammeGeneral,
) -> dict[int, tuple[sp.Expr, sp.Expr]]:
    """`(P(a), P(b))` de chaque variable de base non entiere, au sommet initial."""
    standard = forme_standard(probleme)
    point = noeud(probleme).point or ()
    base = tuple(j for j, v in enumerate(point, start=1) if v != 0)
    tableau = tableau_de_la_base(standard, base)
    return {j: penalites(tableau, j) for j in base if not point[j - 1].is_integer}


def plus_petite_penalite(probleme: ProgrammeGeneral) -> tuple[int, str, sp.Expr]:
    """La variable, la branche (`a` : vers le bas) et la valeur de la plus petite."""
    candidates = [
        (valeur, variable, branche)
        for variable, couple in penalites_a_la_racine(probleme).items()
        for branche, valeur in zip("ab", couple, strict=True)
    ]
    valeur, variable, branche = min(candidates)
    return variable, branche, valeur


def borne_par_la_penalite(evaluation: sp.Expr, penalite: sp.Expr) -> sp.Expr:
    """L'evaluation d'un fils vaut au moins celle du pere plus la penalite."""
    return sp.sympify(evaluation) + sp.sympify(penalite)


def optimum_entier(probleme: ProgrammeGeneral) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """La valeur maximale et le point, par separation et evaluation."""
    entieres = frozenset(range(len(probleme.objectif)))
    exploration = resoudre_en_variables_entieres(ProgrammeEntier(probleme, entieres))
    if exploration.valeur is None or exploration.solution is None:
        msg = "aucun point entier admissible : le probleme est impossible"
        raise ValueError(msg)
    return exploration.valeur, tuple(exploration.solution)


def _plafond(probleme: ProgrammeGeneral, variable: int) -> int:
    """La plus grande valeur que cette variable peut prendre.

    Seule une contrainte dont tous les coefficients sont positifs borne
    une variable a elle seule : les autres termes ne peuvent que retirer.
    """
    bornes = [
        c.second_membre / c.coefficients[variable]
        for c in probleme.contraintes
        if c.coefficients[variable] > 0 and all(a >= 0 for a in c.coefficients)
    ]
    if not bornes:
        msg = f"aucune contrainte ne borne x{variable + 1} a elle seule"
        raise ValueError(msg)
    return int(min(bornes))


def boite(probleme: ProgrammeGeneral) -> list[tuple[int, int]]:
    """Pour chaque variable, de zero a son plafond : ou chercher les points entiers."""
    return [(0, _plafond(probleme, j)) for j in range(len(probleme.objectif))]


def meilleur_point_entier(
    probleme: ProgrammeGeneral,
) -> tuple[sp.Expr, tuple[tuple[int, ...], ...]]:
    """La meilleure valeur et tous les points entiers qui l'atteignent."""
    points = points_entiers(probleme, boite(probleme))
    meilleure = max(probleme.valeur(p) for p in points)
    gagnants = sorted(p for p in points if probleme.valeur(p) == meilleure)
    return meilleure, tuple(gagnants)
