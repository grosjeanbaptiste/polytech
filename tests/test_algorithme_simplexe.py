"""Chapitre II.4 et II.6 d'optimisation : l'algorithme, et les regles contre le cyclage.

Oracles : les illustrations du cours
(`Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf`) -- II.4.4 a)
page 57 (optimum 45 en trois tableaux), II.4.4 b) page 58 (pas de
solution finie), et l'exemple degenere des pages 66 a 69, sur lequel le
cours applique la regle lexicographique (`x4` sort) et enonce celle de
Bland.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import (
    Issue,
    Regle,
    phase_deux,
    pivoter,
)
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.regles import (
    entrante_de_bland,
    sortante_de_bland,
    sortante_lexicographique,
)
from polytech.optimisation.domain.tableau import TableauSimplexe

MAX, MIN = Optimisation.MAX, Optimisation.MIN


def _a1() -> TableauSimplexe:
    coefficients = sp.Matrix([[0, -6, -5], [8, 1, 1], [6, -2, 3], [2, 1, -1]])
    return TableauSimplexe((3, 4, 5), (1, 2), coefficients)


def _b1() -> TableauSimplexe:
    coefficients = sp.Matrix([[0, 3, -5], [6, -2, 3], [4, 1, -4]])
    return TableauSimplexe((3, 4), (1, 2), coefficients)


def _degenere() -> TableauSimplexe:
    coefficients = sp.Matrix([[0, 1, -1], [4, 2, -1], [2, 1, -2], [5, 1, 1]])
    return TableauSimplexe((3, 4, 5), (1, 2), coefficients)


def test_l_illustration_a_s_arrete_a_l_optimum_en_trois_tableaux() -> None:
    resolution = phase_deux(_a1(), MAX)
    assert resolution.issue is Issue.OPTIMALE
    assert len(resolution.tableaux) == 3
    assert resolution.tableaux[-1].valeur == 45
    assert list(resolution.tableaux[-1].solution(5)) == [5, 3, 0, 7, 0]


def test_les_bases_visitees_sont_celles_du_cours() -> None:
    """Points 1, 5 puis 4 de la figure I.12."""
    bases = [t.base for t in phase_deux(_a1(), MAX).tableaux]
    assert bases == [(3, 4, 5), (3, 4, 1), (2, 4, 1)]


def test_la_fonction_economique_s_ameliore_strictement_sans_degenerescence() -> None:
    """Propriete II.4.1 : c'est ce qui fait converger l'algorithme."""
    valeurs = [t.valeur for t in phase_deux(_a1(), MAX).tableaux]
    assert valeurs == [0, 12, 45]


def test_l_illustration_b_n_a_pas_de_solution_optimale_finie() -> None:
    resolution = phase_deux(_b1(), MIN)
    assert resolution.issue is Issue.NON_BORNEE
    assert len(resolution.tableaux) == 2
    assert resolution.colonne_non_bornee == 2


def test_un_tableau_deja_optimal_ne_pivote_pas() -> None:
    optimal = phase_deux(_a1(), MAX).tableaux[-1]
    assert phase_deux(optimal, MAX).tableaux == (optimal,)


def test_l_exemple_degenere_atteint_son_optimum() -> None:
    """Pages 66-67 : `x1 = 2`, `x5 = 3`, `z = -2`."""
    resolution = phase_deux(_degenere(), MIN)
    assert resolution.issue is Issue.OPTIMALE
    assert resolution.tableaux[-1].valeur == -2
    assert list(resolution.tableaux[-1].solution(5)) == [2, 0, 0, 0, 3]


def test_la_regle_lexicographique_fait_sortir_x4() -> None:
    """Page 69 : `a_32 / a_31 = -1/2 > a_42 / a_41 = -2`, donc `x4` sort."""
    assert sortante_lexicographique(_degenere(), entrante=1, variables=5) == 4


def test_la_regle_de_bland_fait_entrer_le_plus_petit_indice_ameliorant() -> None:
    """`k = min{j | a_0j > 0}` : pas le plus grand coefficient, le premier."""
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, 1, 5], [4, 1, 1]]))
    assert entrante_de_bland(tableau, MIN) == 1


def test_la_regle_de_bland_fait_sortir_le_plus_petit_indice_a_egalite() -> None:
    """Sur l'exemple degenere : `x3` et `x4` a egalite, Bland retient `x3`."""
    assert sortante_de_bland(_degenere(), entrante=1) == 3


def test_avec_la_regle_de_bland_l_exemple_degenere_atteint_le_meme_optimum() -> None:
    resolution = phase_deux(_degenere(), MIN, Regle.BLAND)
    assert resolution.tableaux[-1].valeur == -2
    assert list(resolution.tableaux[-1].solution(5)) == [2, 0, 0, 0, 3]


def test_un_plafond_d_iterations_atteint_est_une_erreur_et_non_un_optimum() -> None:
    """L'illustration a) demande deux pivots : un seul tour ne suffit pas."""
    with pytest.raises(RuntimeError, match="aucun arret apres"):
        phase_deux(_a1(), MAX, iterations_maximales=1)


def test_la_regle_de_bland_refuse_un_tableau_optimal() -> None:
    optimal = phase_deux(_a1(), MAX).tableaux[-1]
    with pytest.raises(ValueError, match="aucune variable entrante"):
        entrante_de_bland(optimal, MAX)


def test_les_deux_regles_ne_suivent_pas_le_meme_chemin() -> None:
    """Bland sort `x3` et s'arrete ; la regle lexicographique sort `x4` et repivote."""
    bland = phase_deux(_degenere(), MIN, Regle.BLAND)
    lexicographique = phase_deux(_degenere(), MIN)
    assert [t.base for t in bland.tableaux] == [(3, 4, 5), (1, 4, 5)]
    assert [t.base for t in lexicographique.tableaux] == [
        (3, 4, 5),
        (3, 1, 5),
        (2, 1, 5),
    ]


def test_la_regle_lexicographique_compare_des_rapports_et_non_des_coefficients() -> (
    None
):
    """`x3` : `(4, 2, 3)`, `x4` : `(2, 1, 2)`. Egalite `4/2 = 2/1`, puis `3/2 < 2/1`."""
    tableau = TableauSimplexe(
        (3, 4), (1, 2), sp.Matrix([[0, 1, 0], [4, 2, 3], [2, 1, 2]])
    )
    assert sortante_lexicographique(tableau, entrante=1, variables=4) == 3


def test_pivoter_fait_une_seule_iteration() -> None:
    """Du premier au deuxieme tableau de la page 57, et pas plus loin."""
    suivant = pivoter(_a1(), MAX, Regle.PLUS_FORT_COEFFICIENT)
    assert suivant.base == (3, 4, 1)
    assert suivant.valeur == 12
