"""Chapitre V.1 et V.2 d'optimisation : la phase II de l'algorithme dual.

Oracles (`Optimisation/SLIDES/Slides_05_Lalgorithme_dual_simplexe.pdf`) :
illustration V.2.4 a), pages 117-118 -- `min 6 x1 + 3 x2 + 2 x3`, trois
tableaux recopies coefficient par coefficient ; illustration b), page 118
-- le dual de l'illustration II.4.4 a), trois tableaux, et page 119 la
solution primale `(5, 3)` relue sous les colonnes d'ecart.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import Issue, Regle
from polytech.optimisation.domain.algorithme_dual import (
    phase_deux_duale,
    pivoter_en_dual,
)
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres_duaux import (
    est_dual_admissible,
    est_dual_degenere,
    est_primal_admissible,
    ligne_sans_solution_admissible,
    rapports_duaux,
    variable_entrante_duale,
    variable_sortante_duale,
)
from polytech.optimisation.domain.grand_m import GRAND_M
from polytech.optimisation.domain.tableau import TableauSimplexe

R = sp.Rational
MIN, MAX = Optimisation.MIN, Optimisation.MAX


def _a1() -> TableauSimplexe:
    """Page 117 : la base `{x4, x5}`, duale admissible, non primale admissible."""
    coefficients = sp.Matrix([[0, -6, -3, -2], [-6, -1, -1, -1], [-9, -2, 0, 2]])
    return TableauSimplexe((4, 5), (1, 2, 3), coefficients)


def _b1() -> TableauSimplexe:
    """Page 118 : `min 8 u1 + 6 u2 + 2 u3`, base `{u4, u5}`."""
    coefficients = sp.Matrix([[0, -8, -6, -2], [-6, -1, 2, -1], [-5, -1, -3, 1]])
    return TableauSimplexe((4, 5), (1, 2, 3), coefficients)


def test_la_base_des_ecarts_est_duale_admissible_sans_etre_primale_admissible() -> None:
    assert est_dual_admissible(_a1(), MIN)
    assert not est_primal_admissible(_a1())


def test_en_maximisation_une_base_duale_admissible_a_ses_a0j_non_negatifs() -> None:
    """V.2.3, remarque 1."""
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, 2, 0], [-1, -1, -1]]))
    assert est_dual_admissible(tableau, MAX)
    assert not est_dual_admissible(tableau, MIN)


def test_la_variable_sortante_est_la_plus_negative() -> None:
    """Page 117 : « x5 sort de la base : -9 = min(-6, -9) »."""
    assert variable_sortante_duale(_a1()) == 5


def test_la_variable_entrante_realise_le_plus_petit_rapport() -> None:
    """Page 117 : « -2 est le seul pivot possible » ; puis `-3/-1 = min(6, 3, 4)`."""
    assert variable_entrante_duale(_a1(), sortante=5) == 1
    deuxieme = phase_deux_duale(_a1(), MIN).tableaux[1]
    assert variable_entrante_duale(deuxieme, sortante=4) == 2


def test_les_trois_tableaux_de_l_illustration_a() -> None:
    resolution = phase_deux_duale(_a1(), MIN)
    assert resolution.issue is Issue.OPTIMALE
    assert list(resolution.tableaux) == [
        _a1(),
        TableauSimplexe(
            (4, 1),
            (5, 2, 3),
            sp.Matrix(
                [
                    [27, -3, -3, -8],
                    [R(-3, 2), R(-1, 2), -1, -2],
                    [R(9, 2), R(-1, 2), 0, -1],
                ]
            ),
        ),
        TableauSimplexe(
            (2, 1),
            (5, 4, 3),
            sp.Matrix(
                [
                    [R(63, 2), R(-3, 2), -3, -2],
                    [R(3, 2), R(1, 2), -1, 2],
                    [R(9, 2), R(-1, 2), 0, -1],
                ]
            ),
        ),
    ]


def test_la_fonction_economique_se_degrade_a_chaque_iteration() -> None:
    """Page 113 : `z` croit (0, 27, 63/2) -- c'est `w` qui s'ameliore."""
    valeurs = [t.valeur for t in phase_deux_duale(_a1(), MIN).tableaux]
    assert valeurs == [0, 27, R(63, 2)]


def test_les_trois_tableaux_de_l_illustration_b() -> None:
    resolution = phase_deux_duale(_b1(), MIN)
    assert [t.coefficients for t in resolution.tableaux] == [
        _b1().coefficients,
        sp.Matrix([[12, -6, -10, -2], [6, 1, -2, -1], [-11, -2, -1, 1]]),
        sp.Matrix(
            [
                [45, -3, -7, -5],
                [R(1, 2), R(1, 2), R(-5, 2), R(-1, 2)],
                [R(11, 2), R(-1, 2), R(1, 2), R(-1, 2)],
            ]
        ),
    ]
    final = resolution.tableaux[-1]
    assert (final.base, final.hors_base) == ((3, 1), (5, 2, 4))


def test_la_solution_primale_se_relit_sous_les_colonnes_d_ecart() -> None:
    """Page 119 : `-3 = a_05 = -x2`, `-5 = a_04 = -x1`, d'ou `(5, 3)`."""
    final = phase_deux_duale(_b1(), MIN).tableaux[-1]
    sous = dict(zip(final.hors_base, final.coefficients[0, 1:], strict=True))
    assert (-sous[4], -sous[5]) == (5, 3)


def test_une_ligne_negative_sans_pivot_negatif_rend_le_primal_impossible() -> None:
    """V.1, cas 2.a : `a_l0 < 0` et `a_lj >= 0` pour tout `j`."""
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, -1, -1], [-2, 1, 0]]))
    assert ligne_sans_solution_admissible(tableau) == 3
    resolution = phase_deux_duale(tableau, MIN)
    assert resolution.issue is Issue.IMPOSSIBLE
    assert resolution.tableaux == (tableau,)


def test_une_ligne_negative_qui_a_un_pivot_n_arrete_rien() -> None:
    assert ligne_sans_solution_admissible(_a1()) is None


def test_une_base_deja_primale_admissible_est_optimale_sans_iteration() -> None:
    final = phase_deux_duale(_a1(), MIN).tableaux[-1]
    assert est_primal_admissible(final)
    assert phase_deux_duale(final, MIN).tableaux == (final,)


def test_une_base_non_duale_admissible_est_refusee() -> None:
    """La phase II suppose la condition 2 ; sinon c'est la phase I (V.3)."""
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, 2, -1], [-1, -1, -1]]))
    with pytest.raises(ValueError, match="base non duale admissible"):
        phase_deux_duale(tableau, MIN)


def test_en_maximisation_le_rapport_se_prend_en_valeur_absolue() -> None:
    """V.2.3, remarque 1 : `|a_0k / a_lk| = min |a_0j / a_lj|`."""
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, 6, 2], [-4, -2, -1]]))
    assert variable_entrante_duale(tableau, sortante=3) == 2
    resolution = phase_deux_duale(tableau, MAX)
    assert resolution.issue is Issue.OPTIMALE
    assert resolution.tableaux[-1].base == (2,)


def test_sans_pivot_negatif_il_n_y_a_pas_de_variable_entrante() -> None:
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, -1, -1], [-2, 1, 0]]))
    with pytest.raises(ValueError, match="aucun pivot negatif"):
        variable_entrante_duale(tableau, sortante=3)


def test_sans_variable_negative_il_n_y_a_pas_de_variable_sortante() -> None:
    final = phase_deux_duale(_a1(), MIN).tableaux[-1]
    with pytest.raises(ValueError, match="aucune variable de base negative"):
        variable_sortante_duale(final)


def test_avec_un_grand_m_la_plus_negative_est_celle_du_terme_en_m() -> None:
    """Page 122 : `-9 - 2M` est plus negatif que `-6 + M`, qui est positif."""
    coefficients = sp.Matrix(
        [
            [-2 * GRAND_M, -6, -3, -2],
            [-6 + GRAND_M, -1, -1, 1],
            [-9 - 2 * GRAND_M, -2, 0, -2],
            [GRAND_M, 0, 0, 1],
        ]
    )
    tableau = TableauSimplexe((4, 5, 3), (1, 2, 6), coefficients)
    assert not est_primal_admissible(tableau)
    assert variable_sortante_duale(tableau) == 5
    assert variable_entrante_duale(tableau, sortante=5) == 6


def test_un_a0j_nul_hors_base_signale_la_degenerescence_duale() -> None:
    """V.5.1 : la solution duale est alors a l'intersection de plus de m hyperplans."""
    assert not est_dual_degenere(_a1())
    degenere = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, 0, -1], [-2, -1, -1]]))
    assert est_dual_degenere(degenere)


def test_la_regle_de_bland_duale_prend_les_plus_petits_indices() -> None:
    """V.5.1 : `l = min{i | a_i0 < 0}`, `k = min{p | rapport minimal}`."""
    tableau = TableauSimplexe(
        (4, 5),
        (3, 2, 1),
        sp.Matrix([[0, -2, -2, -4], [-1, -1, -1, -1], [-9, -1, -1, -1]]),
    )
    suivant = phase_deux_duale(tableau, MIN, Regle.BLAND).tableaux[1]
    assert suivant.base == (2, 5)
    plus_negative = phase_deux_duale(tableau, MIN).tableaux[1]
    assert plus_negative.base == (4, 3)


def test_un_plafond_d_iterations_atteint_est_une_erreur_et_non_un_resultat() -> None:
    with pytest.raises(RuntimeError, match="iterations duales"):
        phase_deux_duale(_a1(), MIN, iterations_maximales=1)


def test_une_variable_de_base_nulle_n_est_pas_negative() -> None:
    """Une base degeneree reste primale admissible : `a_i0 >= 0`."""
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, -1, -1], [0, -1, -1]]))
    assert est_primal_admissible(tableau)
    assert phase_deux_duale(tableau, MIN).tableaux == (tableau,)


def test_a_rapports_egaux_la_regle_lexicographique_lit_les_lignes_triviales() -> None:
    """V.5.1 : le tableau est complete des `x_j = x_j`, lignes dans l'ordre 1..n.

    `x2` et `x3` donnent le meme rapport 2. Ligne de `x1` : 0 pour les deux.
    Ligne de `x2` : `|1 / -1|` pour `x2`, 0 pour `x3` -- c'est donc `x3` qui
    entre, bien que `x2` soit ecrite la premiere.
    """
    tableau = TableauSimplexe(
        (4,), (2, 3, 1), sp.Matrix([[0, -2, -2, -4], [-9, -1, -1, -1]])
    )
    assert variable_entrante_duale(tableau, sortante=4) == 3


def test_les_rapports_ne_portent_que_sur_les_pivots_negatifs() -> None:
    """Page 117, second tableau : `min(-3/(-1/2), -3/-1, -8/-2)`, soit 6, 3 et 4."""
    deuxieme = phase_deux_duale(_a1(), MIN).tableaux[1]
    assert rapports_duaux(deuxieme, sortante=4) == {5: 6, 2: 3, 3: 4}
    assert rapports_duaux(_a1(), sortante=5) == {1: 3}


def test_une_iteration_seule_rend_le_tableau_suivant() -> None:
    """Bland sort `x4` (plus petit indice negatif) ; rapports 6, 3, 2 : `x3` entre."""
    assert pivoter_en_dual(_a1()) == phase_deux_duale(_a1(), MIN).tableaux[1]
    assert pivoter_en_dual(_a1(), Regle.BLAND).base == (3, 5)
