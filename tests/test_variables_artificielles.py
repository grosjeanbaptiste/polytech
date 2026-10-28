"""Chapitre II.5 d'optimisation : la phase I, methode des variables artificielles.

Oracles : les deux illustrations II.5.4 du cours
(`Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf`, pages 62 a
65), tableau par tableau -- a) une seule variable artificielle suffit,
l'optimum est `z = 122` en `(12, 10)` ; b) le probleme est impossible,
une variable artificielle reste positive. La variable artificielle `v`
porte l'indice qui suit ceux du probleme.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.domain.algorithme import Issue, Regle, pivoter
from polytech.optimisation.domain.artificielles import (
    chasser_les_nulles,
    sans_colonnes,
    somme_des_artificielles,
)
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres import (
    colonne_sans_solution_finie,
    colonnes_reduisant_la_penalite,
)
from polytech.optimisation.domain.depart import (
    a_seconds_membres_non_negatifs,
    base_evidente,
    probleme_augmente,
    reduire_a_une_artificielle,
)
from polytech.optimisation.domain.formes import FormeStandard
from polytech.optimisation.domain.grand_m import GRAND_M as M
from polytech.optimisation.domain.phase_un import methode_des_variables_artificielles
from polytech.optimisation.domain.tableau import TableauSimplexe, tableau_de_la_base

R = sp.Rational
MAX, MIN = Optimisation.MAX, Optimisation.MIN


def _illustration_a() -> FormeStandard:
    """`max 6 x1 + 5 x2`, `x1 + x2 >= 8`, `-2 x1 + 3 x2 <= 6`, `x1 - x2 <= 2`."""
    return FormeStandard(
        couts=sp.Matrix([[6, 5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, -1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([8, 6, 2]),
    )


def _illustration_b() -> FormeStandard:
    """`x1 + x2 <= 8`, `-2 x1 + 3 x2 >= 6`, `x1 - x2 >= 2` : `D` est vide."""
    return FormeStandard(
        couts=sp.Matrix([[6, 5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 1, 0, 0], [-2, 3, 0, -1, 0], [1, -1, 0, 0, -1]]),
        seconds_membres=sp.Matrix([8, 6, 2]),
    )


def test_un_second_membre_negatif_se_retourne() -> None:
    """« Supposons `d >= 0`, ce qui n'est pas restrictif » (page 59)."""
    forme = FormeStandard(sp.Matrix([[1, 1]]), sp.Matrix([[1, -1]]), sp.Matrix([-3]))
    retournee = a_seconds_membres_non_negatifs(forme)
    assert retournee.matrice == sp.Matrix([[-1, 1]])
    assert retournee.seconds_membres == sp.Matrix([3])


def test_les_ecarts_positifs_forment_une_base_evidente() -> None:
    """Page 63 : `x4` et `x5` servent de base ; la contrainte (1) n'a rien."""
    assert base_evidente(_illustration_a()) == {2: 4, 3: 5}


def test_une_seule_variable_artificielle_suffit_a_l_illustration_a() -> None:
    """Page 63 : `v` dans la contrainte (1), penalisee par `-M` en maximisation."""
    augmente, base = probleme_augmente(_illustration_a(), MAX)
    assert base == (6, 4, 5)
    assert list(augmente.couts) == [6, 5, 0, 0, 0, -M]
    assert list(augmente.matrice[:, 5]) == [1, 0, 0]


def test_en_minimisation_la_penalite_est_positive() -> None:
    augmente, _ = probleme_augmente(_illustration_a(), MIN)
    assert augmente.couts[0, 5] == M


def test_les_tableaux_de_l_illustration_a_sont_ceux_du_cours() -> None:
    tableaux = methode_des_variables_artificielles(_illustration_a(), MAX).tableaux
    assert tableaux[0].coefficients == sp.Matrix(
        [[-8 * M, -M - 6, -M - 5, M], [8, 1, 1, -1], [6, -2, 3, 0], [2, 1, -1, 0]]
    )
    assert tableaux[1].coefficients == sp.Matrix(
        [
            [12 - 6 * M, 6 + M, -11 - 2 * M, M],
            [6, -1, 2, -1],
            [10, 2, 1, 0],
            [2, 1, -1, 0],
        ]
    )
    assert tableaux[2].coefficients == sp.Matrix(
        [
            [45, R(1, 2), -R(11, 2)],
            [3, -R(1, 2), -R(1, 2)],
            [7, R(5, 2), R(1, 2)],
            [5, R(1, 2), -R(1, 2)],
        ]
    )
    assert tableaux[3].coefficients == sp.Matrix(
        [[122, 28, 11], [10, 2, 1], [14, 5, 2], [12, 3, 1]]
    )


def test_la_colonne_de_la_variable_artificielle_sortie_est_supprimee() -> None:
    """Remarque 4 beta : « il est certain qu'elle n'y rentrera plus »."""
    tableaux = methode_des_variables_artificielles(_illustration_a(), MAX).tableaux
    assert tableaux[1].hors_base == (5, 2, 3)
    assert tableaux[2].hors_base == (5, 3)


def test_l_illustration_a_atteint_l_optimum_du_cours() -> None:
    """Page 64 : `x1 = 12`, `x2 = 10`, `x3 = 14`, `z = 122`."""
    resolution = methode_des_variables_artificielles(_illustration_a(), MAX)
    assert resolution.issue is Issue.OPTIMALE
    assert resolution.tableaux[-1].valeur == 122
    assert list(resolution.tableaux[-1].solution(5)) == [12, 10, 14, 0, 0]


def test_l_illustration_b_est_impossible() -> None:
    """Page 65 : une variable artificielle reste positive, `v2 = 14/5`."""
    resolution = methode_des_variables_artificielles(_illustration_b(), MAX)
    assert resolution.issue is Issue.IMPOSSIBLE
    final = resolution.tableaux[-1]
    assert 7 in final.base
    assert final.coefficients[final.base.index(7) + 1, 0] == R(14, 5)


def test_un_probleme_non_borne_le_reste_apres_la_phase_i() -> None:
    """`min -x1`, `x1 - x2 - x3 = 1` : `x1` croit sans fin avec `x2`."""
    forme = FormeStandard(
        sp.Matrix([[-1, 0, 0]]), sp.Matrix([[1, -1, -1]]), sp.Matrix([1])
    )
    assert methode_des_variables_artificielles(forme, MIN).issue is Issue.NON_BORNEE


def test_une_contrainte_redondante_est_revelee_et_eliminee() -> None:
    """Remarque 2 : une artificielle nulle sans pivot possible, `r(T) = m - 1`."""
    forme = FormeStandard(
        couts=sp.Matrix([[1, 0]]),
        matrice=sp.Matrix([[1, 1], [2, 2]]),
        seconds_membres=sp.Matrix([2, 4]),
    )
    resolution = methode_des_variables_artificielles(forme, MIN)
    assert resolution.issue is Issue.OPTIMALE
    assert resolution.contraintes_redondantes == 1
    assert list(resolution.tableaux[-1].solution(2)) == [0, 2]


def test_une_artificielle_nulle_restee_en_base_en_est_chassee() -> None:
    """`x1 + x2 = 1`, `x1 + 2 x2 = 1` : le sommet `(1, 0)` est degenere."""
    forme = FormeStandard(
        couts=sp.Matrix([[0, 1]]),
        matrice=sp.Matrix([[1, 1], [1, 2]]),
        seconds_membres=sp.Matrix([1, 1]),
    )
    resolution = methode_des_variables_artificielles(forme, MIN)
    final = resolution.tableaux[-1]
    assert resolution.issue is Issue.OPTIMALE
    assert resolution.contraintes_redondantes == 0
    assert set(final.base) <= {1, 2}
    assert list(final.solution(2)) == [1, 0]


def test_un_probleme_qui_a_deja_une_base_n_ajoute_aucune_variable() -> None:
    """Toutes les contraintes en `<=` : les ecarts suffisent, la phase I est vide."""
    forme = FormeStandard(
        couts=sp.Matrix([[6, 5, 0, 0, 0]]),
        matrice=sp.Matrix([[1, 1, 1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([8, 6, 2]),
    )
    resolution = methode_des_variables_artificielles(forme, MAX)
    assert resolution.artificielles == ()
    assert resolution.tableaux[-1].valeur == 45


def test_un_pivot_non_nul_chasse_une_artificielle_nulle_sans_changer_z() -> None:
    """Remarque 2 : `x_i^a | 0 | a_ij != 0` ; on pivote, `z` et la solution restent."""
    tableau = TableauSimplexe((3, 1), (2,), sp.Matrix([[1, 5], [0, 2], [1, 1]]))
    nettoye, eliminees = chasser_les_nulles(tableau, artificielles=(3,))
    assert nettoye.base == (2, 1)
    assert nettoye.hors_base == ()
    assert eliminees == 0
    assert nettoye.valeur == tableau.valeur
    assert list(nettoye.solution(2)) == [1, 0]


def test_une_ligne_sans_pivot_possible_est_eliminee() -> None:
    """Remarque 2 : `x_i^a | 0 | 0` ; la contrainte etait redondante."""
    tableau = TableauSimplexe((3, 1), (2,), sp.Matrix([[1, 5], [0, 0], [1, 1]]))
    nettoye, eliminees = chasser_les_nulles(tableau, artificielles=(3,))
    assert nettoye.base == (1,)
    assert eliminees == 1
    assert nettoye.coefficients == sp.Matrix([[1, 5], [1, 1]])


def test_la_somme_des_artificielles_est_nulle_quand_p_a_une_solution() -> None:
    """Remarque 1 : `xi = somme(x^a) = 0`, la solution est admissible pour `P`."""
    resolution = methode_des_variables_artificielles(_illustration_a(), MAX)
    assert (
        somme_des_artificielles(resolution.tableaux[-1], resolution.artificielles) == 0
    )


def test_la_somme_des_artificielles_reste_positive_quand_p_est_impossible() -> None:
    """Remarque 1 : `xi > 0`, le probleme initial est impossible ; ici `14/5`."""
    resolution = methode_des_variables_artificielles(_illustration_b(), MAX)
    xi = somme_des_artificielles(resolution.tableaux[-1], resolution.artificielles)
    assert xi == R(14, 5)


def test_une_seule_artificielle_suffit_quand_les_contraintes_sont_des_inegalites() -> (
    None
):
    """Remarque 3 : `x1 + x2 >= 8` ecrit `-x1 - x2 + x3 = -8` ; on retourne la ligne."""
    ecarts = FormeStandard(
        couts=sp.Matrix([[6, 5, 0, 0, 0]]),
        matrice=sp.Matrix([[-1, -1, 1, 0, 0], [-2, 3, 0, 1, 0], [1, -1, 0, 0, 1]]),
        seconds_membres=sp.Matrix([-8, 6, 2]),
    )
    reduit = reduire_a_une_artificielle(ecarts)
    assert reduit.matrice == _illustration_a().matrice
    assert reduit.seconds_membres == _illustration_a().seconds_membres


def test_les_autres_lignes_negatives_se_soustraient_de_la_plus_negative() -> None:
    """`d = (-3, -5)` : `p = 2` ; la ligne 1 devient `ligne 1 - ligne 2`, `d = 2`."""
    forme = FormeStandard(
        couts=sp.Matrix([[1, 0, 0]]),
        matrice=sp.Matrix([[-1, 1, 0], [-2, 0, 1]]),
        seconds_membres=sp.Matrix([-3, -5]),
    )
    reduit = reduire_a_une_artificielle(forme)
    assert reduit.matrice == sp.Matrix([[1, 1, -1], [2, 0, -1]])
    assert reduit.seconds_membres == sp.Matrix([2, 5])
    assert base_evidente(reduit) == {1: 2}


def test_sans_second_membre_negatif_rien_ne_change() -> None:
    assert reduire_a_une_artificielle(_illustration_a()) == _illustration_a()


def test_sans_colonnes_retire_les_variables_hors_base_designees() -> None:
    """Remarque 4 beta : la colonne de l'artificielle sortie devient superflue."""
    tableau = TableauSimplexe((1,), (2, 3, 4), sp.Matrix([[0, 1, 2, 3], [5, 6, 7, 8]]))
    reduit = sans_colonnes(tableau, (3,))
    assert reduit.hors_base == (2, 4)
    assert reduit.coefficients == sp.Matrix([[0, 1, 3], [5, 6, 8]])


def test_un_probleme_augmente_non_borne_ne_prouve_pas_que_p_le_soit() -> None:
    """Le troisieme couple de la page 95 du chapitre IV contredit la page 60.

    `max 2 x1 + 2 x2`, `-x1 + x2 >= 1`, `x1 - x2 >= 1` est impossible. Son
    probleme augmente, lui, n'est pas borne : le long de `x1 = x2`, `z`
    croit sans fin tandis que les deux variables artificielles restent a
    1. Le cours affirme page 60 qu'alors « les variables artificielles
    sont necessairement toutes nulles » et que `P` est non borne ; c'est
    faux ici. Tant qu'une artificielle est positive, seuls les termes en
    `M` decident, et si aucun ne peut plus la reduire, `P` est impossible.
    """
    forme = FormeStandard(
        couts=sp.Matrix([[2, 2, 0, 0]]),
        matrice=sp.Matrix([[-1, 1, -1, 0], [1, -1, 0, -1]]),
        seconds_membres=sp.Matrix([1, 1]),
    )
    resolution = methode_des_variables_artificielles(forme, MAX)
    assert resolution.issue is Issue.IMPOSSIBLE
    assert (
        somme_des_artificielles(resolution.tableaux[-1], resolution.artificielles) > 0
    )


def test_le_probleme_augmente_de_la_page_95_est_non_borne_artificielles_a_1() -> None:
    """`P^a` est non borne : le simplexe y trouve une colonne sans solution finie.

    Le rayon `x1 = x2 = t` garde les deux artificielles a 1 avec `z = 4 t - 2 M`.
    """
    forme = FormeStandard(
        couts=sp.Matrix([[2, 2, 0, 0]]),
        matrice=sp.Matrix([[-1, 1, -1, 0], [1, -1, 0, -1]]),
        seconds_membres=sp.Matrix([1, 1]),
    )
    augmente, base = probleme_augmente(forme, MAX)
    tableau = tableau_de_la_base(augmente, base)
    assert colonne_sans_solution_finie(tableau, MAX) is None
    tableau = pivoter(tableau, MAX, Regle.PLUS_FORT_COEFFICIENT)
    assert colonne_sans_solution_finie(tableau, MAX) is not None
    for t in (0, 5, 100):
        point = sp.Matrix([t, t, 0, 0, 1, 1])
        assert augmente.matrice * point == augmente.seconds_membres
        assert (augmente.couts * point)[0] == 4 * t - 2 * M


def test_les_colonnes_qui_reduisent_la_penalite_ont_un_terme_en_m_favorable() -> None:
    """Page 63, premier tableau : `x1` et `x2` (`-M - 6`, `-M - 5`), pas `x3` (`M`)."""
    tableaux = methode_des_variables_artificielles(_illustration_a(), MAX).tableaux
    assert colonnes_reduisant_la_penalite(tableaux[0], MAX) == (1, 2)
    assert colonnes_reduisant_la_penalite(tableaux[-1], MAX) == ()
