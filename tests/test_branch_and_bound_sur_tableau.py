"""Chapitre XIV.3 d'optimisation : separer et evaluer sur le tableau simplexe.

Oracles (`Optimisation/SLIDES/Slides_14_Introduction_a_la_programmation_`
`lineaire_en_variables_entieres.pdf`, pages 304 a 311) : l'illustration
`min 7 x1 + 3 x2 + 8 x3`, `x1 + x2 + 3 x3 >= 8`, `3 x1 + x2 + x3 >= 5`,
variables entieres. Tous les tableaux des pages 308 a 310, les penalites
et les deux arborescences de la figure XIV.10 sont recopies.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.optimisation.domain.algorithme_dual import phase_deux_duale
from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.branch_and_bound_dual import (
    Reoptimisation,
    resoudre_par_reoptimisation,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.penalites import (
    Regle,
    choisir_la_separation,
    non_integralite,
    penalites,
)
from polytech.optimisation.domain.separation_sur_tableau import (
    Branche,
    NoeudTableau,
    partie_decimale,
    separer_le_tableau,
)
from polytech.optimisation.domain.sondage import Evaluation, Sondage
from polytech.optimisation.domain.tableau import TableauSimplexe

R = sp.Rational
MIN = Optimisation.MIN
ENTIERES = frozenset({1, 2, 3, 4, 5})
OPTIMUM_CONTINU = sp.Matrix(
    [
        [R(45, 2), -3, R(-1, 2), R(-5, 2)],
        [R(3, 2), -1, R(1, 2), R(-1, 2)],
        [R(7, 2), 4, R(-3, 2), R(1, 2)],
    ]
)
ENTIER = sp.Matrix([[23, -4, -1, -3], [1, -2, -2, -1], [5, 1, -3, -1]])


def _depart() -> TableauSimplexe:
    """Page 308 : la base des ecarts `{x4, x5}`, duale admissible."""
    coefficients = sp.Matrix([[0, -7, -3, -8], [-8, -1, -1, -3], [-5, -3, -1, -1]])
    return TableauSimplexe((4, 5), (1, 2, 3), coefficients)


def _racine() -> NoeudTableau:
    return NoeudTableau(phase_deux_duale(_depart(), MIN).tableaux[-1])


def test_la_relaxation_lineaire_suit_les_trois_tableaux_de_la_page_308() -> None:
    premier, deuxieme, troisieme = phase_deux_duale(_depart(), MIN).tableaux
    assert premier == _depart()
    assert (deuxieme.base, deuxieme.hors_base) == ((3, 5), (1, 2, 4))
    assert deuxieme.coefficients == sp.Matrix(
        [
            [R(64, 3), R(-13, 3), R(-1, 3), R(-8, 3)],
            [R(8, 3), R(1, 3), R(1, 3), R(-1, 3)],
            [R(-7, 3), R(-8, 3), R(-2, 3), R(-1, 3)],
        ]
    )
    assert (troisieme.base, troisieme.hors_base) == ((3, 2), (1, 5, 4))
    assert troisieme.coefficients == OPTIMUM_CONTINU
    assert list(troisieme.solution(5)) == [0, R(7, 2), R(3, 2), 0, 0]


def test_la_partie_decimale_est_ce_qui_depasse_le_plus_grand_entier_inferieur() -> None:
    """`a = [a] + f`, `0 <= f < 1` ; y compris pour un nombre negatif."""
    assert partie_decimale(R(7, 2)) == R(1, 2)
    assert partie_decimale(R(-7, 3)) == R(2, 3)
    assert partie_decimale(sp.Integer(4)) == 0


def test_les_penalites_de_la_racine() -> None:
    """Page 308 : `P3a = 1/2`, `P3b = 3/2`, `P2a = 3/8`, `P2b = 1/6`."""
    tableau = _racine().tableau
    assert penalites(tableau, 3) == (R(1, 2), R(3, 2))
    assert penalites(tableau, 2) == (R(3, 8), R(1, 6))


def test_la_branche_basse_remplace_la_ligne_par_celle_de_l_ecart() -> None:
    """Pages 304-305 et 308 : `x3 <= 1`, ligne `x3e [-1/2 | 1, -1/2, 1/2]`."""
    bas, _haut = separer_le_tableau(_racine(), 3)
    assert bas.description == "x_3 <= 1"
    assert list(bas.tableau.coefficients[1, :]) == [R(-1, 2), 1, R(-1, 2), R(1, 2)]
    assert bas.tableau.coefficients[[0, 2], :] == OPTIMUM_CONTINU[[0, 2], :]


def test_la_branche_haute_ne_change_que_le_second_membre() -> None:
    """Pages 305 et 309 : `x2 >= 4`, ligne `x2' [-1/2 | 4, -3/2, 1/2]`."""
    _bas, haut = separer_le_tableau(_racine(), 2)
    assert haut.description == "x_2 >= 4"
    assert list(haut.tableau.coefficients[2, :]) == [R(-1, 2), 4, R(-3, 2), R(1, 2)]


def test_la_reoptimisation_de_x3_inferieur_a_1_est_immediate() -> None:
    """Page 308 : un pivot, et la solution est entiere, `z = 23`."""
    bas, _haut = separer_le_tableau(_racine(), 3)
    resolution = phase_deux_duale(bas.tableau, MIN)
    assert len(resolution.tableaux) == 2
    final = resolution.tableaux[-1]
    assert (final.base, final.hors_base) == ((5, 2), (1, 3, 4))
    assert final.coefficients == ENTIER


def test_la_reoptimisation_de_x2_superieur_a_4_vaut_68_tiers() -> None:
    """Page 309."""
    _bas, haut = separer_le_tableau(_racine(), 2)
    final = phase_deux_duale(haut.tableau, MIN).tableaux[-1]
    assert (final.base, final.hors_base) == ((3, 5), (1, 2, 4))
    assert final.coefficients == sp.Matrix(
        [
            [R(68, 3), R(-13, 3), R(-1, 3), R(-8, 3)],
            [R(4, 3), R(1, 3), R(1, 3), R(-1, 3)],
            [R(1, 3), R(-8, 3), R(-2, 3), R(-1, 3)],
        ]
    )


def test_les_penalites_du_noeud_s1_de_la_page_309() -> None:
    """`P3a = min(13/3, 1/3)`, `P3b = 16/3`, `P5a` impossible, `P5b = 1/3`."""
    _bas, haut = separer_le_tableau(_racine(), 2)
    final = phase_deux_duale(haut.tableau, MIN).tableaux[-1]
    assert penalites(final, 3) == (R(1, 3), R(16, 3))
    assert penalites(final, 5) == (sp.oo, R(1, 3))


def test_la_reoptimisation_de_x2_inferieur_a_3_vaut_183_huitiemes() -> None:
    """Page 310 : `x2e [-1/2 | -4, 3/2, -1/2]`, puis le tableau de `v4`."""
    bas, _haut = separer_le_tableau(_racine(), 2)
    assert list(bas.tableau.coefficients[2, :]) == [R(-1, 2), -4, R(3, 2), R(-1, 2)]
    final = phase_deux_duale(bas.tableau, MIN).tableaux[-1]
    assert (final.base, final.hors_base) == ((3, 1), (2, 5, 4))
    assert final.coefficients == sp.Matrix(
        [
            [R(183, 8), R(-3, 4), R(-13, 8), R(-17, 8)],
            [R(13, 8), R(-1, 4), R(1, 8), R(-3, 8)],
            [R(1, 8), R(-1, 4), R(-3, 8), R(1, 8)],
        ]
    )


def test_une_penalite_minore_la_degradation_de_la_branche() -> None:
    """Page 306 : `P <= z(i+1) - z(i)` ; ici `3/2 <= v(x3 >= 2) - 45/2`."""
    _bas, haut = separer_le_tableau(_racine(), 3)
    valeur = phase_deux_duale(haut.tableau, MIN).tableaux[-1].valeur
    assert R(45, 2) + R(3, 2) == 24 <= valeur


@pytest.mark.parametrize(
    ("regle", "variable", "branche"),
    [
        (Regle.PLUS_GRANDE_PENALITE, 3, Branche.BASSE),
        (Regle.PLUS_PETITE_PENALITE, 2, Branche.HAUTE),
        (Regle.REGLE_3, 3, Branche.BASSE),
    ],
)
def test_les_trois_regles_de_selection_a_la_racine(
    regle: Regle, variable: int, branche: Branche
) -> None:
    """Pages 306 a 310 : `x3 <= 1` deux fois, `x2 >= 4` pour la plus petite."""
    assert choisir_la_separation(_racine().tableau, ENTIERES, regle) == (
        variable,
        branche,
    )


def test_le_maximum_de_non_integralite_ne_departage_pas_la_racine() -> None:
    """Page 311 : `f = 1 - f = 1/2` pour `x2` comme pour `x3`."""
    tableau = _racine().tableau
    assert non_integralite(tableau, 2) == non_integralite(tableau, 3) == R(1, 2)


def test_l_arborescence_de_la_plus_grande_penalite() -> None:
    """Figure XIV.10 a) : trois noeuds, `x3 <= 1` sonde a 23, `x3 >= 2` elague."""
    exploration = resoudre_par_reoptimisation(
        _depart(), ENTIERES, Regle.PLUS_GRANDE_PENALITE
    )
    assert [n.probleme.description for n in exploration.noeuds] == [
        "",
        "x_3 <= 1",
        "x_3 >= 2",
    ]
    assert [n.sondage for n in exploration.noeuds] == [
        None,
        Sondage.OPTIMALITE,
        Sondage.DOMINANCE,
    ]
    assert [n.evaluation for n in exploration.noeuds] == [R(45, 2), 23, 24]
    assert exploration.valeur == 23
    assert exploration.solution is not None
    assert exploration.solution.point == (0, 5, 1, 0, 1)


def test_l_arborescence_de_la_plus_petite_penalite() -> None:
    """Figure XIV.10 b) : cinq noeuds pour le meme optimum."""
    exploration = resoudre_par_reoptimisation(
        _depart(), ENTIERES, Regle.PLUS_PETITE_PENALITE, couts_entiers=True
    )
    assert [n.probleme.description for n in exploration.noeuds] == [
        "",
        "x_2 >= 4",
        "x_3 <= 1",
        "x_3 >= 2",
        "x_2 <= 3",
    ]
    assert [n.sondage for n in exploration.noeuds] == [
        None,
        None,
        Sondage.OPTIMALITE,
        Sondage.DOMINANCE,
        Sondage.DOMINANCE,
    ]
    assert [n.evaluation for n in exploration.noeuds] == [23, 23, 23, 28, 23]
    assert exploration.valeur == 23
    assert exploration.solution is not None
    assert exploration.solution.point == (0, 5, 1, 0, 1)


def test_sans_l_argument_des_couts_entiers_le_noeud_s4_n_est_pas_sonde() -> None:
    """Page 310 : `v4 = 183/8 < 23` ; « dans ce cas la procedure continue »."""
    exploration = resoudre_par_reoptimisation(
        _depart(), ENTIERES, Regle.PLUS_PETITE_PENALITE
    )
    s4 = exploration.noeuds[4]
    assert s4.probleme.description == "x_2 <= 3"
    assert s4.evaluation == R(183, 8)
    assert s4.sondage is None
    assert len(exploration.noeuds) > 5
    assert exploration.valeur == 23


def test_la_reoptimisation_et_la_separation_sur_les_programmes_s_accordent() -> None:
    """Deux chemins independants, un optimum : `(0, 5, 1)` pour 23."""
    contraintes = (
        Contrainte((1, 1, 3), Sens.SUPERIEUR, 8),
        Contrainte((3, 1, 1), Sens.SUPERIEUR, 5),
    )
    programme = ProgrammeGeneral(MIN, (7, 3, 8), contraintes)
    exploration = resoudre_en_variables_entieres(
        ProgrammeEntier(programme, frozenset({0, 1, 2}))
    )
    assert exploration.valeur == 23
    assert exploration.solution == (0, 5, 1)


def test_une_variable_hors_base_ou_deja_entiere_ne_se_separe_pas() -> None:
    entier = NoeudTableau(TableauSimplexe((5, 2), (1, 3, 4), ENTIER))
    with pytest.raises(ValueError, match="est hors base"):
        separer_le_tableau(_racine(), 1)
    with pytest.raises(ValueError, match="est deja entiere"):
        separer_le_tableau(entier, 2)


def test_un_tableau_entier_n_offre_aucune_separation() -> None:
    tableau = TableauSimplexe((5, 2), (1, 3, 4), ENTIER)
    with pytest.raises(ValueError, match="aucune variable astreinte"):
        choisir_la_separation(tableau, ENTIERES, Regle.REGLE_3)


def test_un_noeud_sans_tableau_optimal_ne_se_separe_pas() -> None:
    """Un noeud sonde par impossibilite n'a pas de solution relaxee a separer."""
    procedure = Reoptimisation(ENTIERES, Regle.REGLE_3)
    with pytest.raises(ValueError, match="noeud sans tableau optimal"):
        procedure.separer(_racine(), Evaluation(None, None, admissible=False))


def test_une_separation_repetee_sur_la_meme_variable_compose_les_substitutions() -> (
    None
):
    """`x2 >= 4`, puis sur ce meme `x2` : les bornes s'expriment en `x2` d'origine."""
    _bas, haut = separer_le_tableau(_racine(), 2)
    assert haut.substitutions == {2: (4, 1)}
    assert haut.valeur_d_origine(2, sp.Integer(1)) == 5
    fractionnaire = NoeudTableau(
        TableauSimplexe((2,), (1,), sp.Matrix([[0, -1], [R(1, 2), -1]])),
        {2: (4, 1)},
    )
    bas, plus_haut = separer_le_tableau(fractionnaire, 2)
    assert (bas.description, plus_haut.description) == ("x_2 <= 4", "x_2 >= 5")
    encore_bas, encore_haut = separer_le_tableau(
        NoeudTableau(fractionnaire.tableau, {2: (4, -1)}), 2
    )
    assert (encore_bas.description, encore_haut.description) == ("x_2 >= 4", "x_2 <= 3")


def test_la_branche_haute_d_une_partie_decimale_d_un_tiers() -> None:
    """Page 309, noeud S3 : `x3 = 4/3`, `x3 >= 2` donne `-(1 - f) = -2/3`."""
    _bas, haut = separer_le_tableau(_racine(), 2)
    s1 = NoeudTableau(phase_deux_duale(haut.tableau, MIN).tableaux[-1])
    bas_x3, haut_x3 = separer_le_tableau(s1, 3)
    assert haut_x3.description == "x_3 >= 2"
    assert list(haut_x3.tableau.coefficients[1, :]) == [
        R(-2, 3),
        R(1, 3),
        R(1, 3),
        R(-1, 3),
    ]
    assert list(bas_x3.tableau.coefficients[1, :]) == [
        R(-1, 3),
        R(-1, 3),
        R(-1, 3),
        R(1, 3),
    ]


def test_la_branche_basse_retient_que_la_ligne_porte_desormais_l_ecart() -> None:
    """`x3 = 1 - x3e` : constante 1, signe -1 ; `x3e = 0` rend `x3 = 1`."""
    bas, _haut = separer_le_tableau(_racine(), 3)
    assert bas.substitutions == {3: (1, -1)}
    assert bas.valeur_d_origine(3, sp.Integer(0)) == 1
    assert bas.valeur_d_origine(2, R(7, 2)) == R(7, 2)


def test_seules_les_variables_astreintes_sont_candidates_a_la_separation() -> None:
    """Probleme mixte : si `x2` seule doit etre entiere, `x3 = 3/2` est laisse."""
    choix = choisir_la_separation(
        _racine().tableau, frozenset({2}), Regle.PLUS_GRANDE_PENALITE
    )
    assert choix == (2, Branche.HAUTE)


def test_le_maximum_de_non_integralite_ignore_les_penalites() -> None:
    """Page 307 : la regle 3, ou `f` et `1 - f` remplacent `P(a)` et `P(b)`.

    `x1 = 5/2` est la plus loin d'un entier ; ses penalites (1/200) sont
    pourtant les plus petites, et la regle 3 lui prefere `x2 = 7/4`.
    """
    tableau = TableauSimplexe(
        (1, 2),
        (3, 4),
        sp.Matrix([[0, -1, -1], [R(5, 2), 100, -100], [R(7, 4), 1, -1]]),
    )
    entieres = frozenset({1, 2})
    assert non_integralite(tableau, 2) == R(1, 4)
    assert choisir_la_separation(tableau, entieres, Regle.NON_INTEGRALITE) == (
        1,
        Branche.BASSE,
    )
    assert choisir_la_separation(tableau, entieres, Regle.REGLE_3) == (2, Branche.HAUTE)
