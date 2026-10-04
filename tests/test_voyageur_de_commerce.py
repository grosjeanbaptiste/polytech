"""Chapitre XIV.4.2 et XIV.4.5 (1) d'optimisation : voyageur de commerce, affectation.

Source : `Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`, pages 314-315 et 317. Le cours
donne la formulation (TSP), ses deux ecritures des contraintes de
sous-tour, et la figure XIV.11 -- la solution de (AP) peut se decomposer
en sous-tours. Sans exemple chiffre dans le cours, chaque modele est
confronte a l'enumeration des permutations.
"""

from __future__ import annotations

from itertools import pairwise, permutations, product
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.contraintes import Contrainte, Sens
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.tournees import (
    SousTours,
    affectation,
    arcs,
    sequencement_avec_reglages,
    sous_tours,
    voyageur_de_commerce,
)
from polytech.optimisation.domain.unimodularite import est_totalement_unimodulaire

if TYPE_CHECKING:
    from polytech.optimisation.domain.entiers import ProgrammeEntier

INFINI = 10
DEUX_PAIRES = (
    (0, 1, INFINI, INFINI),
    (1, 0, INFINI, INFINI),
    (INFINI, INFINI, 0, 1),
    (INFINI, INFINI, 1, 0),
)
ASYMETRIQUE = ((0, 2, 9, 10), (1, 0, 6, 4), (15, 7, 0, 8), (6, 3, 12, 0))
TROIS_VILLES = ((0, 2, 9), (1, 0, 6), (15, 7, 0))


def _cout_du_tour(couts: tuple[tuple[int, ...], ...], ordre: tuple[int, ...]) -> int:
    etapes = zip(ordre, (*ordre[1:], ordre[0]), strict=True)
    return sum(couts[i - 1][j - 1] for i, j in etapes)


def _meilleur_tour(couts: tuple[tuple[int, ...], ...]) -> int:
    """Par enumeration des `(n - 1)!` tours qui partent de la ville 1."""
    villes = range(2, len(couts) + 1)
    return min(_cout_du_tour(couts, (1, *reste)) for reste in permutations(villes))


def _points_binaires(probleme: ProgrammeEntier) -> set[tuple[int, ...]]:
    n = len(probleme.programme.objectif)
    return {p for p in product((0, 1), repeat=n) if probleme.est_admissible(p)}


def test_les_arcs_sont_les_couples_de_villes_distinctes() -> None:
    assert arcs(3) == ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))


def test_l_affectation_impose_un_depart_et_une_arrivee_par_ville() -> None:
    """Page 314, contraintes (1) et (2)."""
    probleme = affectation(TROIS_VILLES)
    egalites = [c for c in probleme.programme.contraintes if c.sens is Sens.EGAL]
    assert egalites[0] == Contrainte((1, 1, 0, 0, 0, 0), Sens.EGAL, 1)
    assert egalites[3] == Contrainte((0, 0, 1, 0, 1, 0), Sens.EGAL, 1)
    assert len(egalites) == 6


def test_une_matrice_de_couts_non_carree_est_refusee() -> None:
    with pytest.raises(ValueError, match="matrice des couts non carree"):
        affectation(((0, 1), (1, 0), (2, 2)))


def test_les_sous_ensembles_des_contraintes_de_sous_tour() -> None:
    """Page 314 : `2 <= |Q| <= n - 2` ; pour quatre villes, les six paires."""
    assert [sorted(q) for q in sous_tours(4)] == [
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
    ]
    assert sous_tours(3) == ()


def test_les_solutions_de_l_affectation_sont_les_permutations_sans_point_fixe() -> None:
    """Quatre villes : neuf derangements, dont six tours et trois couples de paires."""
    assert len(_points_binaires(affectation(ASYMETRIQUE))) == 9


@pytest.mark.parametrize("ecriture", list(SousTours))
def test_les_solutions_du_voyageur_sont_les_seuls_tours(ecriture: SousTours) -> None:
    """Les deux ecritures de (3) eliminent les memes trois solutions a sous-tours."""
    points = _points_binaires(voyageur_de_commerce(ASYMETRIQUE, ecriture))
    assert len(points) == 6
    assert points < _points_binaires(affectation(ASYMETRIQUE))


def test_la_matrice_de_l_affectation_est_totalement_unimodulaire() -> None:
    """XIV.1.4 : c'est pourquoi (AP) se resout en variables continues."""
    probleme = affectation(TROIS_VILLES)
    egalites = [c for c in probleme.programme.contraintes if c.sens is Sens.EGAL]
    matrice = [list(c.coefficients) for c in egalites]
    assert est_totalement_unimodulaire(sp.Matrix(matrice))


def test_la_relaxation_lineaire_de_l_affectation_est_deja_entiere() -> None:
    solution = resoudre(affectation(ASYMETRIQUE).relaxation_lineaire())
    assert solution.point is not None
    assert all(x in (0, 1) for x in solution.point)


def test_l_affectation_minore_le_voyageur_de_commerce() -> None:
    """Page 315 : (AP) est une relaxation de (TSP) ; figure XIV.11, deux sous-tours."""
    de_l_affectation = resoudre(affectation(DEUX_PAIRES).relaxation_lineaire()).valeur
    assert de_l_affectation == 4
    assert _meilleur_tour(DEUX_PAIRES) == 1 + INFINI + 1 + INFINI
    assert de_l_affectation < _meilleur_tour(DEUX_PAIRES)


def test_le_voyageur_de_commerce_par_separation_et_par_enumeration() -> None:
    exploration = resoudre_en_variables_entieres(voyageur_de_commerce(ASYMETRIQUE))
    assert exploration.valeur == _meilleur_tour(ASYMETRIQUE) == 21


def test_l_ordonnancement_avec_reglages_est_un_voyageur_de_commerce() -> None:
    """Page 317 : une tache fictive 0, `t_0j = t_j0 = 0` ; `sum p_j` est constant."""
    durees = (5, 3, 4)
    reglages = ((0, 2, 7), (4, 0, 1), (3, 6, 0))
    probleme, constante = sequencement_avec_reglages(durees, reglages)
    assert constante == 12
    assert len(probleme.programme.objectif) == 4 * 3
    par_enumeration = min(
        sum(reglages[a - 1][b - 1] for a, b in pairwise(ordre))
        for ordre in permutations((1, 2, 3))
    )
    exploration = resoudre_en_variables_entieres(probleme)
    assert exploration.valeur == par_enumeration == 3
    assert exploration.valeur + constante == 15


def test_le_cout_de_chaque_trajet_est_lu_dans_le_sens_du_trajet() -> None:
    """`c_ij` pour `i -> j` : la matrice n'est pas symetrique, l'ordre compte."""
    assert affectation(TROIS_VILLES).programme.objectif == (2, 9, 1, 6, 15, 7)
