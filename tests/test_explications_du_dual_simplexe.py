"""Une iteration de l'algorithme dual, dite comme le cours la dit page 117.

Oracle (`Optimisation/SLIDES/Slides_05_Lalgorithme_dual_simplexe.pdf`,
illustration V.2.4 a) : « La variable x5 sort de la base : -9 = min (-6,
-9). La variable x1 rentre dans la base : -2 est le seul pivot possible »,
puis « x4 sort de la base (seule variable negative). La variable x2
rentre dans la base : -3/-1 = min(-3/(-1/2), -3/-1, -8/-2) ».
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.domain.algorithme_dual import phase_deux_duale
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.explications_du_dual_simplexe import (
    SOURCE,
    expliquer_la_resolution_duale,
    expliquer_une_iteration_duale,
)
from polytech.optimisation.domain.tableau import TableauSimplexe

MIN = Optimisation.MIN


def _a1() -> TableauSimplexe:
    coefficients = sp.Matrix([[0, -6, -3, -2], [-6, -1, -1, -1], [-9, -2, 0, 2]])
    return TableauSimplexe((4, 5), (1, 2, 3), coefficients)


def test_la_sortante_est_justifiee_par_la_valeur_la_plus_negative() -> None:
    sortie = expliquer_une_iteration_duale(_a1()).etapes[0]
    assert sortie.affirmation == "x_5 sort de la base"
    assert "-9 = min(-6, -9)" in sortie.justification
    assert sortie.source == f"{SOURCE}, V.2.1, p. 116"


def test_un_seul_pivot_possible_est_dit_tel() -> None:
    entree = expliquer_une_iteration_duale(_a1()).etapes[1]
    assert entree.affirmation == "x_1 entre dans la base"
    assert "-2 est le seul pivot negatif" in entree.justification
    assert entree.source == f"{SOURCE}, V.1, cas 2.b, p. 115"


def test_l_entrante_est_justifiee_par_le_plus_petit_rapport() -> None:
    deuxieme = phase_deux_duale(_a1(), MIN).tableaux[1]
    sortie, entree, pivot = expliquer_une_iteration_duale(deuxieme).etapes
    assert sortie.affirmation == "x_4 sort de la base"
    assert "seule variable de base negative" in sortie.justification
    assert entree.affirmation == "x_2 entre dans la base"
    assert "-3/-1 = min(-3/(-1/2), -3/-1, -8/-2)" in entree.justification
    assert pivot.affirmation == "le pivot est a_42 = -1"


def test_la_conclusion_dit_que_z_se_degrade() -> None:
    """Page 113 : la nouvelle solution fournit une moins bonne valeur de `z`."""
    raisonnement = expliquer_une_iteration_duale(_a1())
    assert raisonnement.conclusion == "z passe de 0 a 27 : il se degrade, w s'ameliore"


def test_l_arret_optimal_est_justifie_par_les_deux_admissibilites() -> None:
    final = phase_deux_duale(_a1(), MIN).tableaux[-1]
    raisonnement = expliquer_une_iteration_duale(final)
    (etape,) = raisonnement.etapes
    assert etape.affirmation == "la solution de base est optimale"
    assert "a_i0 (3/2, 9/2) sont tous non negatifs" in etape.justification
    assert etape.source == f"{SOURCE}, V.1, cas 1, p. 114"
    assert raisonnement.conclusion == "solution optimale : z = 63/2"


def test_l_arret_impossible_nomme_la_ligne_sans_pivot() -> None:
    tableau = TableauSimplexe((3,), (1, 2), sp.Matrix([[0, -1, -1], [-2, 1, 0]]))
    raisonnement = expliquer_une_iteration_duale(tableau)
    (etape,) = raisonnement.etapes
    assert etape.affirmation == "x_3 = -2 est negative et ne peut pas sortir"
    assert "a_3j (1, 0) sont tous non negatifs" in etape.justification
    assert etape.source == f"{SOURCE}, V.1, cas 2.a, p. 115"
    assert raisonnement.conclusion == "probleme impossible : le dual n'est pas borne"


def test_la_resolution_entiere_enchaine_les_iterations() -> None:
    raisonnement = expliquer_la_resolution_duale(_a1(), MIN)
    affirmations = [e.affirmation for e in raisonnement.etapes]
    assert affirmations == [
        "x_5 sort de la base",
        "x_1 entre dans la base",
        "le pivot est a_51 = -2",
        "x_4 sort de la base",
        "x_2 entre dans la base",
        "le pivot est a_42 = -1",
        "la solution de base est optimale",
    ]
    assert raisonnement.conclusion == "solution optimale : z = 63/2"


def test_une_variable_de_base_nulle_n_est_pas_comptee_parmi_les_negatives() -> None:
    """`x4 = 0` n'est pas negative : `x5` est la seule a pouvoir sortir."""
    coefficients = sp.Matrix([[0, -6, -3, -2], [0, -1, -1, -1], [-9, -2, 0, 2]])
    tableau = TableauSimplexe((4, 5), (1, 2, 3), coefficients)
    sortie = expliquer_une_iteration_duale(tableau).etapes[0]
    assert sortie.affirmation == "x_5 sort de la base"
    assert sortie.justification == "seule variable de base negative"
