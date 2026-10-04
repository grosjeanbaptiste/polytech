"""Une iteration du simplexe, expliquee comme le cours l'ecrit a cote de ses tableaux.

Oracle : la colonne « Algorithme » de l'illustration II.4.4 a)
(`Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf`, page 57) --
« La variable x1 rentre dans la base : -6 = min(-6, -5) », « La variable
x5 sort de la base : 2/1 = min(8/1, 2/1) », « Le pivot est 1 », puis au
dernier tableau : « c'est le tableau optimal etant donne que les
coefficients sont tous non negatifs ».
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.domain.algorithme import phase_deux
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.explications_du_simplexe import (
    expliquer_la_resolution,
    expliquer_une_iteration,
)
from polytech.optimisation.domain.tableau import TableauSimplexe

MAX, MIN = Optimisation.MAX, Optimisation.MIN
SOURCE = "Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf"


def _a1() -> TableauSimplexe:
    coefficients = sp.Matrix([[0, -6, -5], [8, 1, 1], [6, -2, 3], [2, 1, -1]])
    return TableauSimplexe((3, 4, 5), (1, 2), coefficients)


def _b1() -> TableauSimplexe:
    return TableauSimplexe(
        (3, 4), (1, 2), sp.Matrix([[0, 3, -5], [6, -2, 3], [4, 1, -4]])
    )


def test_la_variable_entrante_est_justifiee_par_son_coefficient() -> None:
    entree = expliquer_une_iteration(_a1(), MAX).etapes[0]
    assert entree.affirmation == "x_1 entre dans la base"
    assert "-6 = min(-6, -5)" in entree.justification
    assert "maximisation" in entree.justification


def test_la_variable_sortante_est_justifiee_par_le_plus_petit_rapport() -> None:
    sortie = expliquer_une_iteration(_a1(), MAX).etapes[1]
    assert sortie.affirmation == "x_5 sort de la base"
    assert "2/1 = min(8/1, 2/1)" in sortie.justification


def test_le_pivot_est_nomme() -> None:
    raisonnement = expliquer_une_iteration(_a1(), MAX)
    assert raisonnement.etapes[2].affirmation == "le pivot est a_51 = 1"
    assert "z passe de 0 a 12" in raisonnement.conclusion


def test_chaque_etape_cite_la_propriete_qui_la_fonde() -> None:
    entree, sortie, pivot = expliquer_une_iteration(_a1(), MAX).etapes
    assert entree.source == f"{SOURCE}, II.4.2, p. 55"
    assert sortie.source == f"{SOURCE}, propriete II.2.2, p. 50"
    assert pivot.source == f"{SOURCE}, II.3, p. 54"


def test_un_tableau_optimal_s_explique_par_le_test_d_arret() -> None:
    optimal = phase_deux(_a1(), MAX).tableaux[-1]
    raisonnement = expliquer_une_iteration(optimal, MAX)
    assert len(raisonnement.etapes) == 1
    assert "1/2, 11/2" in raisonnement.etapes[0].justification
    assert raisonnement.etapes[0].source == f"{SOURCE}, propriete II.2.3, p. 52"
    assert "optimale" in raisonnement.conclusion


def test_un_probleme_non_borne_s_explique_par_la_colonne_sans_pivot() -> None:
    """Page 58 : « x2 peut rentrer, mais il n'y a aucun pivot positif possible »."""
    suivant = phase_deux(_b1(), MIN).tableaux[-1]
    raisonnement = expliquer_une_iteration(suivant, MIN)
    assert "x_2" in raisonnement.etapes[0].affirmation
    assert "-5, -4" in raisonnement.etapes[0].justification
    assert raisonnement.etapes[0].source == f"{SOURCE}, propriete II.2.1, p. 50"
    assert "pas de solution optimale finie" in raisonnement.conclusion


def test_la_resolution_enchaine_les_iterations() -> None:
    """Deux changements de base, puis l'arret : sept etapes en tout."""
    raisonnement = expliquer_la_resolution(_a1(), MAX)
    assert len(raisonnement.etapes) == 3 + 3 + 1
    assert "optimale" in raisonnement.conclusion
    assert "z = 45" in raisonnement.conclusion


def test_le_test_d_arret_change_de_signe_avec_le_sens_de_l_optimisation() -> None:
    """Page 57 : « tous non negatifs » en maximisation ; page 52 : `a_0j <= 0` sinon."""
    maximum = phase_deux(_a1(), MAX).tableaux[-1]
    assert (
        "tous non negatifs"
        in expliquer_une_iteration(maximum, MAX).etapes[0].justification
    )
    minimum = TableauSimplexe((1,), (2,), sp.Matrix([[-2, -1], [2, 1]]))
    assert (
        "tous non positifs"
        in expliquer_une_iteration(minimum, MIN).etapes[0].justification
    )
