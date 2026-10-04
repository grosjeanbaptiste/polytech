"""Extensions de l'algorithme simplexe, partie II : les variables bornees.

Oracles (`Optimisation/SLIDES/Slides_Extensions_algorithme_simplexe.pdf`,
planches 15 a 22) : `min 6 x1 + 3 x2 - 2 x3`, `x1 + x2 + x3 >= 6`,
`2 x1 - 2 x3 >= 9`, `0 <= x1 <= 5`, `0 <= x2 <= 1`, `0 <= x3 <= 1`. Les
tableaux des planches 17 a 21 sont recopies ; l'optimum est `x = (5, 1/2,
1/2)`, `z = 61/2`, et les variables duales `(3, 5/2, 2, 0, 0)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.algorithme_borne import resoudre_avec_bornes
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.tableau import TableauSimplexe
from polytech.optimisation.domain.variables_bornees import (
    TableauBorne,
    complementer,
    translater_les_bornes_inferieures,
)

R = sp.Rational
MIN = Optimisation.MIN
SUP, INF = Sens.SUPERIEUR, Sens.INFERIEUR
BORNES = {1: 5, 2: 1, 3: 1}


def _depart() -> TableauBorne:
    """Planche 17 : la base des ecarts, ni primale ni duale admissible."""
    coefficients = sp.Matrix([[0, -6, -3, 2], [-6, -1, -1, -1], [-9, -2, 0, 2]])
    return TableauBorne(TableauSimplexe((4, 5), (1, 2, 3), coefficients), BORNES)


def test_le_changement_de_variable_d_une_variable_hors_base() -> None:
    """Planche 18 : `x3 = 1 - x3e` ; `z` part de -2, la base est duale admissible."""
    apres = complementer(_depart(), 3)
    assert apres.complementees == frozenset({3})
    assert apres.tableau.coefficients == sp.Matrix(
        [[-2, -6, -3, -2], [-5, -1, -1, 1], [-11, -2, 0, -2]]
    )


def test_le_changement_de_variable_d_une_variable_de_base() -> None:
    """Planche 19 : `x3e = 11/2 > 1`, d'ou la ligne de `x3` : `[-9/2 | -1 0 1/2]`."""
    coefficients = sp.Matrix(
        [[9, -4, -3, -1], [R(-21, 2), -2, -1, R(1, 2)], [R(11, 2), 1, 0, R(-1, 2)]]
    )
    avant = TableauBorne(
        TableauSimplexe((4, 3), (1, 2, 5), coefficients), BORNES, frozenset({3})
    )
    apres = complementer(avant, 3)
    assert apres.complementees == frozenset()
    assert list(apres.tableau.coefficients[2, :]) == [R(-9, 2), -1, 0, R(1, 2)]
    assert apres.tableau.coefficients[:2, :] == coefficients[:2, :]


def test_le_changement_de_variable_est_involutif() -> None:
    """`x = u - xe` et `xe = u - x` : deux fois de suite, on revient au depart."""
    assert complementer(complementer(_depart(), 3), 3) == _depart()


def test_une_variable_sans_borne_superieure_ne_se_complemente_pas() -> None:
    with pytest.raises(ValueError, match="sans borne superieure"):
        complementer(_depart(), 4)


def test_les_tableaux_des_planches_18_a_21() -> None:
    resolution = resoudre_avec_bornes(_depart(), MIN)
    assert resolution.issue is Issue.OPTIMALE
    lus = [
        (e.tableau.base, e.tableau.hors_base, e.complementees)
        for e in resolution.etapes
    ]
    assert lus == [
        ((4, 5), (1, 2, 3), frozenset()),
        ((4, 5), (1, 2, 3), frozenset({3})),
        ((4, 3), (1, 2, 5), frozenset({3})),
        ((4, 3), (1, 2, 5), frozenset()),
        ((1, 3), (4, 2, 5), frozenset()),
        ((1, 3), (4, 2, 5), frozenset({1})),
        ((2, 3), (4, 1, 5), frozenset({1})),
    ]
    assert resolution.etapes[4].tableau.coefficients == sp.Matrix(
        [
            [30, -2, -1, -2],
            [R(21, 4), R(-1, 2), R(1, 2), R(-1, 4)],
            [R(3, 4), R(-1, 2), R(1, 2), R(1, 4)],
        ]
    )
    assert resolution.etapes[5].tableau.coefficients[1, :] == sp.Matrix(
        [[R(-1, 4), R(1, 2), R(-1, 2), R(1, 4)]]
    )
    assert resolution.etapes[6].tableau.coefficients == sp.Matrix(
        [
            [R(61, 2), -3, -2, R(-5, 2)],
            [R(1, 2), -1, -2, R(-1, 2)],
            [R(1, 2), 0, 1, R(1, 2)],
        ]
    )


def test_la_solution_optimale_de_la_planche_21() -> None:
    """`x1 = 5` (`x1e = 0`), `x2 = 1/2`, `x3 = 1/2`."""
    resolution = resoudre_avec_bornes(_depart(), MIN)
    assert list(resolution.solution(5)) == [5, R(1, 2), R(1, 2), 0, 0]
    assert resolution.valeur == R(61, 2) == 6 * 5 + 3 * R(1, 2) - 2 * R(1, 2)


def test_les_variables_duales_de_la_planche_22() -> None:
    """`u1 = 3`, `u2 = 5/2` sous les ecarts ; `u3 = 2` pour la borne de `x1`."""
    resolution = resoudre_avec_bornes(_depart(), MIN)
    assert resolution.variables_duales((4, 5)) == (3, R(5, 2))
    assert resolution.duales_des_bornes() == {1: 2, 2: 0, 3: 0}


def test_les_bornes_traitees_en_contraintes_donnent_le_meme_optimum() -> None:
    """Planche 16 : l'autre voie, qui double le nombre de variables."""
    lignes = (
        Contrainte((1, 1, 1), SUP, 6),
        Contrainte((2, 0, -2), SUP, 9),
        Contrainte((1, 0, 0), INF, 5),
        Contrainte((0, 1, 0), INF, 1),
        Contrainte((0, 0, 1), INF, 1),
    )
    solution = resoudre(ProgrammeGeneral(MIN, (6, 3, -2), lignes))
    assert solution.point == (5, R(1, 2), R(1, 2))
    assert solution.valeur == R(61, 2)


def test_sans_les_bornes_l_optimum_est_celui_de_la_planche_20() -> None:
    """« C'est le tableau optimal sans les contraintes de borne superieure » : 30."""
    lignes = (Contrainte((1, 1, 1), SUP, 6), Contrainte((2, 0, -2), SUP, 9))
    solution = resoudre(ProgrammeGeneral(MIN, (6, 3, -2), lignes))
    assert solution.valeur == 30
    assert solution.point == (R(21, 4), 0, R(3, 4))


def test_un_probleme_borne_impossible_est_reconnu() -> None:
    """`x1 >= 6` avec `x1 <= 5` : la ligne negative n'a aucun pivot negatif."""
    coefficients = sp.Matrix([[0, -1], [-6, -1]])
    depart = TableauBorne(TableauSimplexe((2,), (1,), coefficients), {1: 5})
    assert resoudre_avec_bornes(depart, MIN).issue is Issue.IMPOSSIBLE


def test_une_base_qu_aucune_borne_ne_rend_duale_admissible_est_refusee() -> None:
    """`a_04 > 0` et `x4` n'a pas de borne : la methode ne s'applique pas."""
    coefficients = sp.Matrix([[0, 2, -1], [-1, -1, -1]])
    depart = TableauBorne(TableauSimplexe((3,), (4, 1), coefficients), {1: 5})
    with pytest.raises(ValueError, match="sans borne superieure"):
        resoudre_avec_bornes(depart, MIN)


def test_les_bornes_inferieures_se_ramenent_a_zero_par_translation() -> None:
    """Planche 15 : `x' = x - l` ; `z' = c x' + c l`, `T x' >= d - T l`."""
    programme = ProgrammeGeneral(
        MIN, (6, 3, -2), (Contrainte((1, 1, 1), SUP, 6), Contrainte((2, 0, -2), SUP, 9))
    )
    translate, constante = translater_les_bornes_inferieures(programme, (1, 0, 2))
    assert constante == 6 * 1 + 3 * 0 - 2 * 2
    assert translate.objectif == programme.objectif
    assert translate.contraintes == (
        Contrainte((1, 1, 1), SUP, 6 - 3),
        Contrainte((2, 0, -2), SUP, 9 - (2 - 4)),
    )


def test_la_translation_conserve_la_valeur_de_z_en_tout_point() -> None:
    programme = ProgrammeGeneral(MIN, (6, 3, -2), (Contrainte((1, 1, 1), SUP, 6),))
    translate, constante = translater_les_bornes_inferieures(programme, (1, 0, 2))
    x = (4, 2, 3)
    x_prime = (3, 2, 1)
    assert translate.valeur(x_prime) + constante == programme.valeur(x)
    assert translate.contraintes[0].ecart(x_prime) == programme.contraintes[0].ecart(x)


def test_un_plafond_d_iterations_atteint_est_une_erreur_et_non_un_resultat() -> None:
    with pytest.raises(RuntimeError, match="aucun arret de l'algorithme a variables"):
        resoudre_avec_bornes(_depart(), MIN, iterations_maximales=2)


def test_une_variable_exactement_a_sa_borne_la_verifie() -> None:
    """`x_i <= u_i` : l'egalite n'appelle aucun changement de variable."""
    coefficients = sp.Matrix([[5, -1], [5, 1]])
    depart = TableauBorne(TableauSimplexe((1,), (2,), coefficients), {1: 5})
    resolution = resoudre_avec_bornes(depart, MIN)
    assert resolution.issue is Issue.OPTIMALE
    assert resolution.etapes == (depart,)


def test_une_variable_nulle_hors_base_ne_charge_pas_sa_borne() -> None:
    """Planche 22 : `x_i < u_i` entraine une variable duale nulle pour la borne.

    `min x1 + 2 x2`, `x1 + x2 >= 1`, bornes 5 et 5. A l'optimum `(1, 0)`,
    `x2` est hors base avec `a_02 = -1` : ce coefficient est le prix de sa
    non-negativite, non celui de sa borne superieure.
    """
    coefficients = sp.Matrix([[0, -1, -2], [-1, -1, -1]])
    depart = TableauBorne(TableauSimplexe((3,), (1, 2), coefficients), {1: 5, 2: 5})
    resolution = resoudre_avec_bornes(depart, MIN)
    assert list(resolution.solution(3)) == [1, 0, 0]
    assert resolution.duales_des_bornes() == {1: 0, 2: 0}
