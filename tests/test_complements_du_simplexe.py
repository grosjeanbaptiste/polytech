"""Chapitre II.7 d'optimisation : toutes les solutions optimales, lire le tableau.

Oracles (`Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf`) :
page 71, illustration alpha -- `max 5 x1 + 5 x2` a deux sommets optimaux,
`(5, 3)` et `(18/5, 22/5)`, de valeur 40 ; illustration beta --
`min -2 x1 + 8 x2` a un sommet optimal et un rayon de solutions
optimales ; page 73 -- les valeurs duales `11/2`, `0`, `1/2` lues sur le
tableau optimal de l'illustration II.4.4 a).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import phase_deux
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.interpretation import (
    effet_d_une_unite,
    gain_de_l_iteration,
    valeurs_marginales,
)
from polytech.optimisation.domain.optimales import (
    autres_sommets_optimaux,
    colonnes_des_rayons_optimaux,
    est_l_unique_optimum,
)
from polytech.optimisation.domain.tableau import TableauSimplexe

R = sp.Rational
MAX, MIN = Optimisation.MAX, Optimisation.MIN


def _optimal(objectif: tuple[int, int]) -> TableauSimplexe:
    """Le tableau optimal de `max c1 x1 + c2 x2` sur le polygone de la figure I.6."""
    c1, c2 = objectif
    depart = sp.Matrix([[0, -c1, -c2], [8, 1, 1], [6, -2, 3], [2, 1, -1]])
    return phase_deux(TableauSimplexe((3, 4, 5), (1, 2), depart), MAX).tableaux[-1]


def _beta() -> TableauSimplexe:
    """Page 71 : `min -2 x1 + 8 x2`, memes contraintes que l'illustration II.4.4 b)."""
    depart = sp.Matrix([[0, 2, -8], [6, -2, 3], [4, 1, -4]])
    return phase_deux(TableauSimplexe((3, 4), (1, 2), depart), MIN).tableaux[-1]


def test_des_coefficients_tous_stricts_donnent_un_optimum_unique() -> None:
    """Propriete II.7.1 : `a_0j < 0` pour tout `j` hors base (en minimisation)."""
    assert est_l_unique_optimum(_optimal((6, 5)), MAX)


def test_un_coefficient_nul_laisse_la_place_a_d_autres_optimums() -> None:
    assert _optimal((5, 5)).coefficients[0, :] == sp.Matrix([[40, 0, 5]])
    assert not est_l_unique_optimum(_optimal((5, 5)), MAX)


def test_l_autre_sommet_optimal_de_l_illustration_alpha() -> None:
    """Page 71 : `x1 = 18/5`, `x2 = 22/5`, `x5 = 14/5`, et toujours `z = 40`."""
    (autre,) = autres_sommets_optimaux(_optimal((5, 5)))
    assert autre.valeur == 40
    assert list(autre.solution(5)) == [R(18, 5), R(22, 5), 0, 0, R(14, 5)]


def test_un_optimum_unique_n_a_pas_d_autre_sommet() -> None:
    assert autres_sommets_optimaux(_optimal((6, 5))) == ()


def test_l_illustration_beta_a_un_rayon_de_solutions_optimales() -> None:
    """Page 71 : `x1 = 4`, `z = -8` ; augmenter `x2` ne change pas `z`, sans borne."""
    tableau = _beta()
    assert tableau.valeur == -8
    assert list(tableau.solution(4)) == [4, 0, 14, 0]
    assert colonnes_des_rayons_optimaux(tableau) == (2,)
    assert autres_sommets_optimaux(tableau) == ()


def test_les_valeurs_duales_se_lisent_sous_les_colonnes_des_ecarts() -> None:
    """Page 73 : `u1 = a_03 = 11/2`, `u2 = a_04 = 0`, `u3 = a_05 = 1/2`."""
    assert valeurs_marginales(_optimal((6, 5)), (3, 4, 5)) == (R(11, 2), 0, R(1, 2))


def test_une_unite_d_une_activite_hors_base_deplace_la_solution_et_le_cout() -> None:
    """Page 72 : `x_i = a_i0 - a_ik` et `z = a_00 - a_0k`, pour `x_k = 1`."""
    tableau = _optimal((6, 5))
    valeur, niveaux = effet_d_une_unite(tableau, activite=3)
    assert valeur == 45 - R(11, 2)
    assert niveaux == {2: 3 - R(1, 2), 4: 7 + R(1, 2), 1: 5 - R(1, 2)}


def test_le_gain_d_une_iteration_est_a0k_fois_le_plus_petit_rapport() -> None:
    """Propriete II.2.2 : `z* = a_00 - a_0k a_l0 / a_lk` ; ici `0 - (-6)(2/1) = 12`."""
    depart = sp.Matrix([[0, -6, -5], [8, 1, 1], [6, -2, 3], [2, 1, -1]])
    tableau = TableauSimplexe((3, 4, 5), (1, 2), depart)
    assert gain_de_l_iteration(tableau, entrante=1) == 12


def test_une_iteration_sans_pivot_n_a_pas_de_gain_fini() -> None:
    """Page 58 : `x2` peut entrer, aucun pivot ne le borne."""
    tableau = TableauSimplexe(
        (3, 1), (4, 2), sp.Matrix([[-12, -3, 7], [14, 2, -5], [4, 1, -4]])
    )
    with pytest.raises(ValueError, match="aucun pivot positif"):
        gain_de_l_iteration(tableau, entrante=2)


def test_une_arete_bornee_n_est_pas_un_rayon() -> None:
    """Illustration alpha : l'arete optimale relie deux sommets."""
    assert colonnes_des_rayons_optimaux(_optimal((5, 5))) == ()
