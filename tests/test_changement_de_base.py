"""Chapitre II.2 et II.3 d'optimisation : les trois proprietes, et le pivot.

Oracles : les tableaux du cours
(`Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf`) --
illustration a) page 57 (maximisation, trois tableaux), illustration b)
page 58 (minimisation, pas de solution finie), illustration de la
degenerescence page 67 (deux variables peuvent sortir).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres import (
    colonne_sans_solution_finie,
    est_optimal,
    variable_entrante,
    variables_sortantes,
)
from polytech.optimisation.domain.pivot import changer_de_base
from polytech.optimisation.domain.tableau import TableauSimplexe

R = sp.Rational
MAX, MIN = Optimisation.MAX, Optimisation.MIN


def _a1() -> TableauSimplexe:
    """Premier tableau de la page 57 : `max 6 x1 + 5 x2`."""
    coefficients = sp.Matrix([[0, -6, -5], [8, 1, 1], [6, -2, 3], [2, 1, -1]])
    return TableauSimplexe((3, 4, 5), (1, 2), coefficients)


def _a3() -> TableauSimplexe:
    return changer_de_base(changer_de_base(_a1(), sortante=5, entrante=1), 3, 2)


def _b1() -> TableauSimplexe:
    """Premier tableau de la page 58 : `min -3 x1 + 5 x2`."""
    return TableauSimplexe(
        (3, 4), (1, 2), sp.Matrix([[0, 3, -5], [6, -2, 3], [4, 1, -4]])
    )


def _degenere() -> TableauSimplexe:
    """Premier tableau de la page 67 : `min -x1 + x2`."""
    coefficients = sp.Matrix([[0, 1, -1], [4, 2, -1], [2, 1, -2], [5, 1, 1]])
    return TableauSimplexe((3, 4, 5), (1, 2), coefficients)


def test_le_premier_pivot_donne_le_deuxieme_tableau_du_cours() -> None:
    """`x1` entre, `x5` sort, pivot 1 : `x5` prend la place de `x1`."""
    suivant = changer_de_base(_a1(), sortante=5, entrante=1)
    assert suivant.base == (3, 4, 1)
    assert suivant.hors_base == (5, 2)
    assert suivant.coefficients == sp.Matrix(
        [[12, 6, -11], [6, -1, 2], [10, 2, 1], [2, 1, -1]]
    )


def test_le_second_pivot_donne_le_tableau_optimal_du_cours() -> None:
    """`x2` entre, `x3` sort, pivot 2."""
    assert _a3().base == (2, 4, 1)
    assert _a3().hors_base == (5, 3)
    assert _a3().coefficients == sp.Matrix(
        [
            [45, R(1, 2), R(11, 2)],
            [3, -R(1, 2), R(1, 2)],
            [7, R(5, 2), -R(1, 2)],
            [5, R(1, 2), R(1, 2)],
        ]
    )


def test_les_quatre_regles_du_changement_de_base() -> None:
    """Page 54, sur le pivot 2 du deuxieme tableau (`x3` sort, `x2` entre)."""
    avant = changer_de_base(_a1(), sortante=5, entrante=1).coefficients
    apres = _a3().coefficients
    pivot = avant[1, 2]
    assert apres[1, 2] == 1 / pivot
    assert apres[1, 1] == avant[1, 1] / pivot
    assert apres[2, 2] == -avant[2, 2] / pivot
    assert apres[2, 1] == avant[2, 1] - avant[2, 2] * avant[1, 1] / pivot


def test_un_zero_dans_la_ligne_du_pivot_fige_sa_colonne() -> None:
    """Page 54 : si `a_lj = 0`, les `a_ij` de la colonne `j` ne changent pas."""
    tableau = TableauSimplexe(
        (3, 4), (1, 2), sp.Matrix([[0, 2, 3], [4, 1, 0], [6, 2, 5]])
    )
    suivant = changer_de_base(tableau, sortante=3, entrante=1)
    assert suivant.coefficients[:, 2] == tableau.coefficients[:, 2]


def test_un_pivot_nul_est_refuse() -> None:
    tableau = TableauSimplexe(
        (3, 4), (1, 2), sp.Matrix([[0, 2, 3], [4, 0, 1], [6, 2, 5]])
    )
    with pytest.raises(ValueError, match="pivot nul"):
        changer_de_base(tableau, sortante=3, entrante=1)


def test_une_variable_qui_n_est_pas_en_base_ne_peut_pas_sortir() -> None:
    with pytest.raises(ValueError, match="n'est pas en base"):
        changer_de_base(_a1(), sortante=1, entrante=2)


def test_en_maximisation_le_coefficient_le_plus_negatif_entre() -> None:
    """Page 57 : `x1` rentre, `-6 = min(-6, -5)`."""
    assert variable_entrante(_a1(), MAX) == 1


def test_le_plus_petit_rapport_designe_la_variable_sortante() -> None:
    """Page 57 : `x5` sort, `2/1 = min(8/1, 2/1)`."""
    assert variables_sortantes(_a1(), entrante=1) == (5,)


def test_le_tableau_final_de_la_maximisation_est_optimal() -> None:
    """Page 57 : les coefficients `1/2` et `11/2` sont tous non negatifs."""
    assert est_optimal(_a3(), MAX)
    assert not est_optimal(_a1(), MAX)


def test_en_minimisation_le_plus_grand_coefficient_positif_entre() -> None:
    """Page 58 : `x1` correspond au seul coefficient `a_0j` positif."""
    assert variable_entrante(_b1(), MIN) == 1
    assert variables_sortantes(_b1(), entrante=1) == (4,)


def test_une_colonne_sans_pivot_positif_dit_le_probleme_non_borne() -> None:
    """Page 58 : `x2` peut rentrer, mais ses coefficients valent -5 et -4."""
    suivant = changer_de_base(_b1(), sortante=4, entrante=1)
    assert suivant.coefficients == sp.Matrix([[-12, -3, 7], [14, 2, -5], [4, 1, -4]])
    assert colonne_sans_solution_finie(suivant, MIN) == 2
    assert colonne_sans_solution_finie(_b1(), MIN) is None


def test_un_tableau_optimal_n_a_aucune_variable_entrante() -> None:
    with pytest.raises(ValueError, match="aucune variable entrante"):
        variable_entrante(_a3(), MAX)


def test_deux_variables_peuvent_sortir_quand_les_rapports_sont_egaux() -> None:
    """Page 67 : `4/2 = 2/1`, `x3` et `x4` peuvent egalement sortir."""
    assert variable_entrante(_degenere(), MIN) == 1
    assert variables_sortantes(_degenere(), entrante=1) == (3, 4)


def test_sortir_l_une_ou_l_autre_donne_une_solution_degeneree() -> None:
    """Page 67 : `x4 = 0` si `x3` sort, `x3 = 0` si `x4` sort ; meme sommet."""
    par_x3 = changer_de_base(_degenere(), sortante=3, entrante=1)
    par_x4 = changer_de_base(_degenere(), sortante=4, entrante=1)
    assert par_x3.coefficients == sp.Matrix(
        [
            [-2, -R(1, 2), -R(1, 2)],
            [2, R(1, 2), -R(1, 2)],
            [0, -R(1, 2), -R(3, 2)],
            [3, -R(1, 2), R(3, 2)],
        ]
    )
    assert par_x4.coefficients == sp.Matrix(
        [[-2, -1, 1], [0, -2, 3], [2, 1, -2], [3, -1, 3]]
    )
    assert list(par_x3.solution(5)) == list(par_x4.solution(5)) == [2, 0, 0, 0, 3]


def test_un_coefficient_nul_n_empeche_pas_l_optimalite() -> None:
    """Page 71 : `z | 40 | 0  5`, tableau optimal de `max 5 x1 + 5 x2`."""
    coefficients = sp.Matrix(
        [
            [40, 0, 5],
            [3, -R(1, 2), R(1, 2)],
            [7, R(5, 2), -R(1, 2)],
            [5, R(1, 2), R(1, 2)],
        ]
    )
    assert est_optimal(TableauSimplexe((2, 4, 1), (5, 3), coefficients), MAX)


def test_la_variable_entrante_n_est_pas_toujours_la_premiere_colonne() -> None:
    """Page 57, deuxieme tableau : `x2` rentre, `-11` est le seul negatif."""
    deuxieme = changer_de_base(_a1(), sortante=5, entrante=1)
    assert deuxieme.hors_base == (5, 2)
    assert variable_entrante(deuxieme, MAX) == 2
    assert variables_sortantes(deuxieme, entrante=2) == (3,)


def test_une_colonne_sans_pivot_mais_sans_attrait_ne_dit_rien() -> None:
    """Faire croitre `x1` irait a l'infini, mais degraderait `z` : on ne le fait pas."""
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, -1, 2], [1, -1, 1]]))
    assert colonne_sans_solution_finie(tableau, MIN) is None
