"""`intervalles_de_definition` de tan et cot : intervalle d'etude borne."""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as X
from polytech.maths.domain.differentielles.prolongement import (
    est_globale,
    intervalles_de_definition,
)

PI = sp.pi
OUVERT = sp.Interval.open


def test_tan_sur_deux_periodes_a_quatre_poles() -> None:
    attendus = (
        OUVERT(-2 * PI, -3 * PI / 2),
        OUVERT(-3 * PI / 2, -PI / 2),
        OUVERT(-PI / 2, PI / 2),
        OUVERT(PI / 2, 3 * PI / 2),
        OUVERT(3 * PI / 2, 2 * PI),
    )
    assert intervalles_de_definition(sp.tan(X), OUVERT(-2 * PI, 2 * PI)) == attendus


def test_cot_a_ses_poles_en_k_pi() -> None:
    attendus = (OUVERT(-PI, 0), OUVERT(0, PI), OUVERT(PI, 2 * PI))
    assert intervalles_de_definition(sp.cot(X), OUVERT(-PI, 2 * PI)) == attendus


def test_tan_de_x_sur_deux_a_son_premier_pole_en_pi() -> None:
    attendus = (OUVERT(-2 * PI, -PI), OUVERT(-PI, PI), OUVERT(PI, 2 * PI))
    assert intervalles_de_definition(sp.tan(X / 2), OUVERT(-2 * PI, 2 * PI)) == attendus


def test_sans_pole_dans_l_intervalle_il_reste_entier() -> None:
    assert intervalles_de_definition(sp.tan(X), OUVERT(-1, 1)) == (OUVERT(-1, 1),)
    assert est_globale(sp.Integer(0))


def test_un_pole_exactement_au_bord_ouvert_ne_coupe_rien() -> None:
    assert intervalles_de_definition(sp.tan(X), OUVERT(-PI / 2, PI / 2)) == (
        OUVERT(-PI / 2, PI / 2),
    )


def test_un_pole_au_bord_ferme_est_retire_du_bord() -> None:
    attendu = (sp.Interval.open(-PI / 2, PI / 2),)
    assert intervalles_de_definition(sp.tan(X), sp.Interval(-PI / 2, PI / 2)) == attendu


def test_tan_sans_intervalle_d_etude_leve() -> None:
    with pytest.raises(ValueError, match="intervalle d'etude"):
        intervalles_de_definition(sp.tan(X))


def test_tan_sur_un_intervalle_non_borne_leve() -> None:
    with pytest.raises(ValueError, match="intervalle d'etude"):
        intervalles_de_definition(sp.tan(X), OUVERT(0, sp.oo))


def test_argument_non_lineaire_leve() -> None:
    with pytest.raises(ValueError, match="argument"):
        intervalles_de_definition(sp.tan(X**2), OUVERT(-1, 1))


def test_l_intervalle_d_etude_restreint_aussi_les_expressions_sans_periode() -> None:
    assert intervalles_de_definition(-1 / (X + 2), OUVERT(-5, 0)) == (
        OUVERT(-5, -2),
        OUVERT(-2, 0),
    )
