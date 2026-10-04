"""Chapitre V.3 et V.4 d'optimisation : la methode de la contrainte artificielle.

Oracles (`Optimisation/SLIDES/Slides_05_Lalgorithme_dual_simplexe.pdf`,
pages 121-122) : illustration a) -- `min 6 x1 + 3 x2 - 2 x3`, `K = {3}`,
cinq tableaux recopies, optimum `z = 30` en `(21/4, 0, 3/4)` ;
illustration b) -- `min -6 x1 + 3 x2 - 2 x3`, `K = {1, 3}`, probleme non
borne (`z = -6M`).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import Issue, Regle
from polytech.optimisation.domain.contrainte_artificielle import (
    ResolutionDuale,
    indices_a_borner,
    methode_de_la_contrainte_artificielle,
    origine_du_rayon,
    probleme_augmente_dual,
)
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.grand_m import GRAND_M as M
from polytech.optimisation.domain.tableau import TableauSimplexe

R = sp.Rational
MIN, MAX = Optimisation.MIN, Optimisation.MAX
CORPS = [[-6, -1, -1, -1], [-9, -2, 0, 2]]
V = 6


def _depart(ligne_de_z: list[int]) -> TableauSimplexe:
    return TableauSimplexe((4, 5), (1, 2, 3), sp.Matrix([ligne_de_z, *CORPS]))


def _a() -> TableauSimplexe:
    """Page 121 : la base des ecarts, ni primale ni duale admissible."""
    return _depart([0, -6, -3, 2])


def _b() -> TableauSimplexe:
    return _depart([0, 6, -3, 2])


def test_k_reunit_les_indices_dont_le_coefficient_viole_la_condition_duale() -> None:
    """`K = {j | a_0j > 0}` : `{3}` pour a), `{1, 3}` pour b)."""
    assert indices_a_borner(_a(), MIN) == (3,)
    assert indices_a_borner(_b(), MIN) == (1, 3)


def test_en_maximisation_k_reunit_les_a0j_negatifs() -> None:
    """V.3.2, remarque 1."""
    assert indices_a_borner(_a(), MAX) == (1, 2)


def test_le_probleme_augmente_recoit_la_ligne_de_la_contrainte_artificielle() -> None:
    """Page 121 : `x3 + v = M`, soit la ligne `v [M | 0 0 1]`."""
    augmente = probleme_augmente_dual(_a(), MIN)
    assert augmente.base == (4, 5, V)
    assert augmente.hors_base == (1, 2, 3)
    assert augmente.coefficients == sp.Matrix([[0, -6, -3, 2], *CORPS, [M, 0, 0, 1]])


def test_la_contrainte_artificielle_de_b_porte_sur_x1_et_x3() -> None:
    """Page 122 : `x1 + x3 + v = M`."""
    assert list(probleme_augmente_dual(_b(), MIN).coefficients[3, :]) == [M, 1, 0, 1]


def test_les_cinq_tableaux_de_l_illustration_a() -> None:
    resolution = methode_de_la_contrainte_artificielle(_a(), MIN)
    assert resolution.issue is Issue.OPTIMALE
    impose, apres_v, sans_v, optimal = resolution.tableaux[1:]
    assert (impose.base, impose.hors_base) == ((4, 5, 3), (1, 2, V))
    assert impose.coefficients == sp.Matrix(
        [
            [-2 * M, -6, -3, -2],
            [-6 + M, -1, -1, 1],
            [-9 - 2 * M, -2, 0, -2],
            [M, 0, 0, 1],
        ]
    )
    assert (apres_v.base, apres_v.hors_base) == ((4, V, 3), (1, 2, 5))
    assert apres_v.coefficients == sp.Matrix(
        [
            [9, -4, -3, -1],
            [R(-21, 2), -2, -1, R(1, 2)],
            [R(9, 2) + M, 1, 0, R(-1, 2)],
            [R(-9, 2), -1, 0, R(1, 2)],
        ]
    )
    assert (sans_v.base, sans_v.hors_base) == ((4, 3), (1, 2, 5))
    assert (optimal.base, optimal.hors_base) == ((1, 3), (4, 2, 5))
    assert optimal.coefficients == sp.Matrix(
        [
            [30, -2, -1, -2],
            [R(21, 4), R(-1, 2), R(1, 2), R(-1, 4)],
            [R(3, 4), R(-1, 2), R(1, 2), R(1, 4)],
        ]
    )


def test_l_optimum_de_a_est_le_point_2_de_la_figure_iv_5() -> None:
    final = methode_de_la_contrainte_artificielle(_a(), MIN).tableaux[-1]
    assert list(final.solution(5)) == [R(21, 4), 0, R(3, 4), 0, 0]
    assert final.valeur == 6 * R(21, 4) - 2 * R(3, 4) == 30


def test_le_premier_changement_de_base_est_impose() -> None:
    """V.3.1 : `v` sort, et entre la variable de `K` au plus grand `a_0j`."""
    resolution = methode_de_la_contrainte_artificielle(_b(), MIN)
    impose = resolution.tableaux[1]
    assert (impose.base, impose.hors_base) == ((4, 5, 1), (V, 2, 3))
    assert impose.coefficients == sp.Matrix(
        [
            [-6 * M, -6, -3, -4],
            [-6 + M, 1, -1, 0],
            [-9 + 2 * M, 2, 0, 4],
            [M, 1, 0, 1],
        ]
    )


def test_un_optimum_qui_depend_de_m_avec_v_hors_base_est_un_probleme_non_borne() -> (
    None
):
    """Page 122, cas 2 du V.3.1 : `z = -6M`."""
    resolution = methode_de_la_contrainte_artificielle(_b(), MIN)
    assert resolution.issue is Issue.NON_BORNEE
    assert len(resolution.tableaux) == 2
    assert resolution.tableaux[-1].valeur == -6 * M


def test_un_probleme_augmente_sans_solution_admissible_rend_p_impossible() -> None:
    """Cas 1 du V.3.1. `min -x1`, `x1 + x2 = -1` : aucun `x >= 0` ne convient."""
    depart = TableauSimplexe((2,), (1,), sp.Matrix([[0, 1], [-1, 1]]))
    resolution = methode_de_la_contrainte_artificielle(depart, MIN)
    assert resolution.issue is Issue.IMPOSSIBLE


def test_une_base_deja_duale_admissible_ne_recoit_pas_de_contrainte() -> None:
    """`K` est vide : la phase I n'a pas lieu d'etre (V.1)."""
    depart = _depart([0, -6, -3, -2])
    resolution = methode_de_la_contrainte_artificielle(depart, MIN)
    assert resolution.tableaux[0] == depart
    assert resolution.tableaux[-1].valeur == R(63, 2)
    with pytest.raises(ValueError, match="deja duale admissible"):
        probleme_augmente_dual(depart, MIN)


def test_un_optimum_independant_de_m_avec_v_hors_base_est_un_rayon_d_optimums() -> None:
    """Cas 2, seconde branche : `z` ne depend pas de `M`.

    `min -x1 + x2`, `x1 - x2 + x3 = 1`. Apres le pivot impose puis l'entree
    de `x2` : `z = -1`, `x1 = M`, `x2 = M - 1`. Le rayon part du point
    obtenu pour le plus petit `M` qui garde `x >= 0` : `M = 1`, `(1, 0)`.
    """
    final = TableauSimplexe(
        (1, 2), (4, 3), sp.Matrix([[-1, 0, -1], [M, 1, 0], [M - 1, 1, -1]])
    )
    assert list(origine_du_rayon(final, variables=3)) == [1, 0, 0]


def test_l_origine_du_rayon_n_est_jamais_prise_pour_un_m_negatif() -> None:
    final = TableauSimplexe((1,), (4, 2), sp.Matrix([[-1, 0, -1], [M + 3, 1, 0]]))
    assert list(origine_du_rayon(final, variables=2)) == [3, 0]


def test_la_methode_reconnait_le_rayon_d_optimums_de_bout_en_bout() -> None:
    """`min -x1 + x2`, `x1 - x2 + x3 = 1` : `z = -1` tout le long de `x1 - x2 = 1`.

    Apres le pivot impose, `x3 = 1 - M` sort ; `v` et `x2` donnent le meme
    rapport 1, et les deux regles du V.5.1 font entrer `x2` (la ligne de
    `x1` vaut 1 sous `v`, 0 sous `x2`). `v` reste hors base et `z` ne
    depend pas de `M` : cas 2, seconde branche.
    """
    depart = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, 1, -1], [1, 1, -1]]))
    for regle in Regle:
        resolution = methode_de_la_contrainte_artificielle(depart, MIN, regle)
        final = resolution.tableaux[-1]
        assert resolution.issue is Issue.OPTIMALE
        assert final.valeur == -1
        assert resolution.artificielle in final.hors_base
        assert list(origine_du_rayon(final, variables=3)) == [1, 0, 0]


def test_un_plafond_d_iterations_atteint_est_une_erreur_et_non_un_resultat() -> None:
    with pytest.raises(RuntimeError, match="aucun arret de la methode"):
        methode_de_la_contrainte_artificielle(_a(), MIN, iterations_maximales=1)


def test_un_coefficient_nul_ne_viole_pas_la_condition_duale() -> None:
    """`a_0j <= 0` : l'egalite est permise, `x_j` n'entre pas dans `K`."""
    assert indices_a_borner(_depart([0, 0, -3, 2]), MIN) == (3,)


def test_quand_v_est_en_base_plus_rien_ne_depend_de_m() -> None:
    """Page 120 : `v` est alors « la seule a dependre de la constante M »."""
    resolution = methode_de_la_contrainte_artificielle(_a(), MIN)
    for tableau in resolution.tableaux[3:]:
        assert not tableau.coefficients.has(M)


def test_la_resolution_nomme_la_variable_artificielle_qu_elle_a_ajoutee() -> None:
    """`v` prend l'indice qui suit le dernier du tableau ; aucune si `K` est vide."""
    avec = methode_de_la_contrainte_artificielle(_a(), MIN)
    sans = methode_de_la_contrainte_artificielle(_depart([0, -6, -3, -2]), MIN)
    assert isinstance(avec, ResolutionDuale)
    assert avec.artificielle == V
    assert sans.artificielle is None
