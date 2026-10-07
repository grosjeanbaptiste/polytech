"""Exemples resolus, planches 63 a 78 : « Branch and Bound » et pretraitement.

L'illustration minimise `7x1 + 3x2 + 8x3` en variables entieres ; les
planches la deroulent selon plusieurs regles de separation. Un second
exemple montre qu'arrondir l'optimum continu ne donne pas l'optimum
entier.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import sympy as sp

from polytech.optimisation.application import simplexe_entiers as arbre
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain import pretraitement
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import points_entiers
from polytech.optimisation.domain.formulation import ProgrammeGeneral

INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR
ILLUSTRATION: Final = ProgrammeGeneral(
    Optimisation.MIN,
    (7, 3, 8),
    (Contrainte((1, 1, 3), SUP, 8), Contrainte((3, 1, 1), SUP, 5)),
)
"""Planche 64."""
ADMISSIBLE: Final = (0, 8, 0)
"""Un point entier admissible evident : il borne ce qu'il reste a chercher."""
AUTRE: Final = ProgrammeGeneral(
    Optimisation.MAX,
    (17, 12),
    (Contrainte((10, 7), INF, 40), Contrainte((1, 1), INF, 5)),
)
"""Planche 75."""
BORNES: Final = ((1, 3), (1, 3), (1, 3))
"""Planches 77 et 78 : chaque variable entiere est entre 1 et 3."""
Optimum = tuple[sp.Expr, tuple[sp.Expr, ...]]


def noeud(*bornes: arbre.Borne) -> arbre.Noeud:
    """La relaxation de l'illustration, ces contraintes de separation ajoutees."""
    return arbre.noeud(ILLUSTRATION, bornes)


def penalites_a_la_racine() -> dict[int, tuple[sp.Expr, sp.Expr]]:
    """`(P(a), P(b))` pour `x2` et `x3`, les deux variables de base."""
    return arbre.penalites_a_la_racine(ILLUSTRATION)


def plus_grande_penalite() -> tuple[int, str, sp.Expr]:
    """La regle des planches 67 a 69."""
    return arbre.penalite_extreme(ILLUSTRATION, grande=True)


def plus_petite_penalite() -> tuple[int, str, sp.Expr]:
    """La regle des planches 70 a 73."""
    return arbre.penalite_extreme(ILLUSTRATION, grande=False)


def borne(evaluation: sp.Expr, penalite: sp.Expr) -> sp.Expr:
    """L'evaluation du pere plus la penalite, arrondie : l'objectif est entier."""
    return sp.ceiling(arbre.borne_par_la_penalite(evaluation, penalite))


def optimum_entier() -> Optimum:
    """La valeur minimale et le point, par separation et evaluation."""
    return arbre.optimum_entier(ILLUSTRATION)


def boite() -> list[tuple[int, int]]:
    """Ou chercher : un point qui vaut au plus `ADMISSIBLE` a `x_j <= 24 / c_j`."""
    plafond = ILLUSTRATION.valeur(ADMISSIBLE)
    return [(0, int(plafond / cout)) for cout in ILLUSTRATION.objectif]


def meilleur_point_entier(
    *bornes: arbre.Borne,
) -> tuple[sp.Expr, tuple[tuple[int, ...], ...]]:
    """Second chemin : le minimum sur tous les points entiers du sous-probleme."""
    ajoutees = tuple(_separation(b) for b in bornes)
    contraintes = (*ILLUSTRATION.contraintes, *ajoutees)
    points = points_entiers(replace(ILLUSTRATION, contraintes=contraintes), boite())
    meilleure = min(ILLUSTRATION.valeur(p) for p in points)
    gagnants = sorted(p for p in points if ILLUSTRATION.valeur(p) == meilleure)
    return meilleure, tuple(gagnants)


def _separation(borne_ajoutee: arbre.Borne) -> Contrainte:
    variable, sens, valeur = borne_ajoutee
    coefficients = [0] * len(ILLUSTRATION.objectif)
    coefficients[variable - 1] = 1
    return Contrainte(tuple(coefficients), sens, valeur)


def autre_exemple() -> tuple[Optimum, Optimum]:
    """Planche 75 : l'optimum continu, puis l'optimum entier."""
    return optimum_fini(AUTRE), arbre.optimum_entier(AUTRE)


def est_admissible_dans_l_autre(point: tuple[int, ...]) -> bool:
    """Vrai si le point respecte les deux contraintes de la planche 75."""
    return all(c.ecart(point) >= 0 for c in AUTRE.contraintes)


def valeur_dans_l_autre(point: tuple[int, ...]) -> sp.Expr:
    """L'objectif `17x1 + 12x2` en ce point."""
    return AUTRE.valeur(point)


def est_redondante(coefficients: tuple[int, ...], second_membre: int) -> bool:
    """Planche 77 : les bornes suffisent a verifier la contrainte."""
    contrainte = Contrainte(coefficients, INF, second_membre)
    return pretraitement.est_redondante(contrainte, BORNES)


def est_contradictoire(coefficients: tuple[int, ...], second_membre: int) -> bool:
    """Planche 77 : aucun point de la boite ne verifie la contrainte."""
    contrainte = Contrainte(coefficients, INF, second_membre)
    return pretraitement.est_impossible(contrainte, BORNES)


def resserree(
    coefficients: tuple[int, ...], second_membre: int
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[sp.Expr, ...], sp.Expr]]:
    """Planche 78 : les bornes resserrees, puis la contrainte simplifiee."""
    contrainte = Contrainte(coefficients, INF, second_membre)
    bornes = pretraitement.resserrer_les_bornes(contrainte, BORNES)
    simplifiee = pretraitement.fixer_les_variables(contrainte, bornes)
    return bornes, (simplifiee.coefficients, simplifiee.second_membre)
