"""Extensions de l'algorithme simplexe, partie I : la forme revisee.

Oracles (`Optimisation/SLIDES/Slides_Extensions_algorithme_simplexe.pdf`,
planches 10 a 13) : `min -6 x1 - 5 x2` sur les contraintes de la figure
I.6. Le tableau des donnees, les deux facteurs `E1` et `E2`, l'inverse
`B2^-1` et les premieres lignes successives sont recopies des planches.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import Issue, phase_deux
from polytech.optimisation.domain.algorithme_revise import (
    resoudre_en_forme_revisee,
)
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.forme_revisee import (
    facteur_etendu,
    inverse_de_base_etendue,
    tableau_des_donnees,
)
from polytech.optimisation.domain.formes import FormeStandard
from polytech.optimisation.domain.tableau import tableau_de_la_base

R = sp.Rational
E1 = sp.Matrix([[1, 0, 0, -6], [0, 1, 0, -1], [0, 0, 1, 2], [0, 0, 0, 1]])
E2 = sp.Matrix(
    [[1, R(-11, 2), 0, 0], [0, R(1, 2), 0, 0], [0, R(-1, 2), 1, 0], [0, R(1, 2), 0, 1]]
)
B2 = sp.Matrix(
    [
        [1, R(-11, 2), 0, R(-1, 2)],
        [0, R(1, 2), 0, R(-1, 2)],
        [0, R(-1, 2), 1, R(5, 2)],
        [0, R(1, 2), 0, R(1, 2)],
    ]
)


def _forme() -> FormeStandard:
    """Planche 10 : le probleme, variables d'ecart comprises."""
    return FormeStandard(
        couts=sp.Matrix([[-6, -5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([8, 6, 2]),
    )


def test_le_tableau_des_donnees_de_la_planche_10() -> None:
    """`[[0, -c, 1], [d, T, 0]]` : la colonne de `z` ferme le tableau."""
    assert tableau_des_donnees(_forme()) == sp.Matrix(
        [
            [0, 6, 5, 0, 0, 0, 1],
            [8, 1, 1, 1, 0, 0, 0],
            [6, -2, 3, 0, 1, 0, 0],
            [2, 1, -1, 0, 0, 1, 0],
        ]
    )


def test_l_inverse_etendue_de_la_base_des_ecarts_est_l_identite() -> None:
    """Planche 10 : `B0 = I = B0^-1`."""
    assert inverse_de_base_etendue(_forme(), (3, 4, 5)) == sp.eye(4)


def test_l_inverse_etendue_porte_cb_b_inverse_sur_sa_premiere_ligne() -> None:
    """Planche 5 : `[[1, c_B B^-1], [0, B^-1]]` ; ici `B2^-1` de la planche 12."""
    assert inverse_de_base_etendue(_forme(), (2, 4, 1)) == B2


def test_l_inverse_etendue_fois_les_donnees_est_le_tableau_simplexe_complet() -> None:
    """Planche 5 : `B^-1 (T, d)` ; confronte au tableau du chapitre II."""
    produit = inverse_de_base_etendue(_forme(), (2, 4, 1)) * tableau_des_donnees(
        _forme()
    )
    complet = tableau_de_la_base(_forme(), (2, 4, 1)).complet(5)
    assert produit[:, :-1] == complet
    assert list(produit[:, -1]) == [1, 0, 0, 0]


def test_les_deux_facteurs_des_planches_11_et_12() -> None:
    """La colonne entrante, etendue de son `a_0l`, et la ligne du pivot."""
    assert facteur_etendu(sp.Matrix([6, 1, -2, 1]), ligne=3) == E1
    assert facteur_etendu(sp.Matrix([11, 2, 1, -1]), ligne=1) == E2


def test_un_pivot_nul_n_a_pas_de_facteur() -> None:
    with pytest.raises(ValueError, match="pivot nul"):
        facteur_etendu(sp.Matrix([6, 0, -2, 1]), ligne=1)


def test_la_resolution_revisee_suit_les_planches_11_a_13() -> None:
    resolution = resoudre_en_forme_revisee(_forme(), (3, 4, 5))
    assert resolution.issue is Issue.OPTIMALE
    premiere, deuxieme, derniere = resolution.iterations
    assert premiere.base == (3, 4, 5)
    assert list(premiere.ligne_de_z) == [0, 6, 5, 0, 0, 0]
    assert (premiere.entrante, premiere.sortante) == (1, 5)
    assert premiere.colonne == sp.Matrix([6, 1, -2, 1])
    assert premiere.facteur == E1
    assert deuxieme.base == (3, 4, 1)
    assert deuxieme.inverse == E1
    assert list(deuxieme.ligne_de_z) == [-12, 0, 11, 0, 0, -6]
    assert list(deuxieme.seconds_membres) == [6, 10, 2]
    assert (deuxieme.entrante, deuxieme.sortante) == (2, 3)
    assert deuxieme.colonne == sp.Matrix([11, 2, 1, -1])
    assert deuxieme.facteur == E2
    assert derniere.base == (2, 4, 1)
    assert derniere.inverse == E2 * E1 == B2
    assert list(derniere.ligne_de_z) == [-45, 0, 0, R(-11, 2), 0, R(-1, 2)]
    assert list(derniere.seconds_membres) == [3, 7, 5]
    assert derniere.facteur is None


def test_la_solution_optimale_de_la_planche_13() -> None:
    resolution = resoudre_en_forme_revisee(_forme(), (3, 4, 5))
    assert list(resolution.solution(5)) == [5, 3, 0, 7, 0]
    assert resolution.valeur == -45


def test_la_solution_duale_se_lit_sur_la_premiere_ligne_de_l_inverse() -> None:
    """Planche 13 : `u = (11/2, 0, 1/2)`, au signe du passage `max` -> `min` pres.

    La planche donne les valeurs duales du probleme `max 6 x1 + 5 x2` du
    chapitre II ; pour `min -6 x1 - 5 x2`, `c_B B^-1` en est l'oppose.
    """
    resolution = resoudre_en_forme_revisee(_forme(), (3, 4, 5))
    assert resolution.solution_duale == (R(-11, 2), 0, R(-1, 2))


def test_la_forme_revisee_visite_les_bases_de_l_algorithme_du_chapitre_ii() -> None:
    """Meme regle, memes bases : seul le calcul differe."""
    revisee = resoudre_en_forme_revisee(_forme(), (3, 4, 5))
    depart = tableau_de_la_base(_forme(), (3, 4, 5))
    classique = phase_deux(depart, Optimisation.MIN)
    assert [i.base for i in revisee.iterations] == [t.base for t in classique.tableaux]
    assert revisee.valeur == classique.tableaux[-1].valeur


def test_un_probleme_non_borne_est_reconnu_sans_tableau_complet() -> None:
    """Illustration II.4.4 b) : la colonne entrante n'a aucun pivot positif."""
    forme = FormeStandard(
        couts=sp.Matrix([[-3, 5, 0, 0]]),
        matrice=sp.Matrix([[-2, 3, 1, 0], [1, -4, 0, 1]]),
        seconds_membres=sp.Matrix([6, 4]),
    )
    resolution = resoudre_en_forme_revisee(forme, (3, 4))
    assert resolution.issue is Issue.NON_BORNEE
    assert resolution.iterations[-1].sortante is None


def test_une_base_non_admissible_est_refusee() -> None:
    with pytest.raises(ValueError, match="base non admissible"):
        resoudre_en_forme_revisee(_forme(), (1, 2, 3))


def test_un_plafond_d_iterations_atteint_est_une_erreur_et_non_un_resultat() -> None:
    with pytest.raises(RuntimeError, match="aucun arret de la forme revisee"):
        resoudre_en_forme_revisee(_forme(), (3, 4, 5), iterations_maximales=2)


def test_une_base_singuliere_est_refusee() -> None:
    """`t3` et `t4` et `t5` moins une : `(1, 2, 2)` repete une colonne."""
    with pytest.raises(ValueError, match="n'est pas une base"):
        inverse_de_base_etendue(_forme(), (1, 2, 2))


def test_a_rapports_egaux_la_premiere_ligne_sort() -> None:
    """Sans regle anti-cyclage ici : la forme revisee ne change que le calcul."""
    forme = FormeStandard(
        couts=sp.Matrix([[-1, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 0], [1, 0, 1]]),
        seconds_membres=sp.Matrix([2, 2]),
    )
    premiere = resoudre_en_forme_revisee(forme, (2, 3)).iterations[0]
    assert premiere.sortante == 2


def test_un_coefficient_nul_dans_la_colonne_entrante_n_est_pas_un_pivot() -> None:
    """Propriete II.2.2 : le rapport ne se prend que sur les `a_ik > 0`."""
    forme = FormeStandard(
        couts=sp.Matrix([[-1, 0, 0]]),
        matrice=sp.Matrix([[0, 1, 0], [1, 0, 1]]),
        seconds_membres=sp.Matrix([0, 2]),
    )
    resolution = resoudre_en_forme_revisee(forme, (2, 3))
    assert resolution.iterations[0].sortante == 3
    assert list(resolution.solution(3)) == [2, 0, 0]
