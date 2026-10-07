"""Modelisation, partie III : l'affectation et le voyageur de commerce.

Deux affirmations des planches 72 et 78 se pretent au calcul : la
matrice des contraintes d'affectation est totalement unimodulaire, et
les variables `y` de la planche 77 rendent tout sous-tour impossible.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

COUTS_D_ESSAI: Final = ((4, 2, 8), (4, 3, 7), (3, 1, 6))
"""Un exemple de ce document : les planches ne chiffrent aucune affectation."""
SOLUTION_AP: Final = {1: 2, 2: 3, 3: 1, 4: 5, 5: 6, 6: 4}
SOLUTION_TSP: Final = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}
"""Planche 78 : le successeur de chaque ville."""


def matrice_d_affectation(taille: int) -> sp.Matrix:
    """Une ligne par ressource, puis une par activite ; une colonne par `x_ij`."""
    paires = list(itertools.product(range(taille), repeat=2))
    ressources = [[int(i == ligne) for i, _ in paires] for ligne in range(taille)]
    activites = [[int(j == ligne) for _, j in paires] for ligne in range(taille)]
    return sp.Matrix(ressources + activites)


def mineurs(matrice: sp.Matrix) -> tuple[int, set[sp.Expr]]:
    """Le nombre de sous-matrices carrees, et l'ensemble de leurs determinants."""
    nombre, valeurs = 0, set()
    for ordre in range(1, min(matrice.shape) + 1):
        lignes = itertools.combinations(range(matrice.rows), ordre)
        for choix in itertools.product(
            lignes, itertools.combinations(range(matrice.cols), ordre)
        ):
            valeurs.add(matrice.extract(list(choix[0]), list(choix[1])).det())
            nombre += 1
    return nombre, valeurs


def affectation_par_le_simplexe(
    couts: Sequence[Sequence[int]],
) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Le probleme d'affectation, les `x_ij` seulement positifs : aucune binaire."""
    matrice = matrice_d_affectation(len(couts))
    contraintes = tuple(
        Contrainte(tuple(matrice.row(i)), Sens.EGAL, 1) for i in range(matrice.rows)
    )
    objectif = tuple(c for ligne in couts for c in ligne)
    return optimum_fini(ProgrammeGeneral(Optimisation.MIN, objectif, contraintes))


def meilleure_permutation(
    couts: Sequence[Sequence[int]],
) -> tuple[int, tuple[int, ...]]:
    """La moins chere des affectations, toutes passees en revue."""
    permutations = itertools.permutations(range(len(couts)))
    return min((sum(couts[i][j] for i, j in enumerate(p)), p) for p in permutations)


def sous_tours(successeurs: Mapping[int, int]) -> tuple[tuple[int, ...], ...]:
    """Les circuits d'une solution, chacun a partir de sa plus petite ville."""
    restantes, circuits = set(successeurs), []
    while restantes:
        ville = min(restantes)
        circuit = []
        while ville in restantes:
            restantes.remove(ville)
            circuit.append(ville)
            ville = successeurs[ville]
        circuits.append(tuple(circuit))
    return tuple(circuits)


def contradiction_du_sous_tour(circuit: Sequence[int]) -> int:
    """`y_j >= y_i + 1` sur chaque arc : sommees, il reste `0 >=` ce nombre."""
    return len(circuit)


def ordre_possible(successeurs: Mapping[int, int]) -> bool:
    """Vrai s'il existe des `y` respectant les contraintes de la planche 77."""
    villes = sorted(successeurs)
    taille = len(villes)
    contraintes = []
    for i, j in successeurs.items():
        if j == villes[0]:
            continue
        ligne = tuple(int(v == j) - int(v == i) for v in villes)
        contraintes.append(Contrainte(ligne, Sens.SUPERIEUR, 1))
    modele = ProgrammeGeneral(Optimisation.MIN, (1,) * taille, tuple(contraintes))
    return resoudre(modele).issue is Issue.OPTIMALE
