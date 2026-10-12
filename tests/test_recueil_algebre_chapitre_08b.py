"""Le chapitre VIII du recueil d'algebre, section VIII.7 : exercices 14 a 32.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 112 a 116 ;
l'oracle est « Solutions du 8eme chapitre », pages 125 a 128 du syllabus.
Matrices associees, changements de base, projections.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch08b_bases as bases
from polytech.maths.application import recueil_algebre_ch08b_matrices as matrices
from polytech.maths.application import recueil_algebre_ch08b_suite as suite
from polytech.maths.application.macros_recueil_algebre_ch08b import (
    macros_de_la_partie_b,
)
from polytech.maths.application.recueil_algebre import nom_de_la_cle
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.sous_espaces import sont_supplementaires

M = sp.Matrix
R = sp.Rational
I = sp.I  # noqa: E741  # l'unite imaginaire, que le syllabus note i
A, B, C, D = matrices.LETTRES
X, Y, Z = sp.symbols("x y z")

ORACLES = {
    "14a": M([5, 3]),
    "14b": M([11, 5, -1]),
    "15a": M([[1, 0, 0, 0], [0, 1, 0, 0]]),
    "16a": M([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]),
    "16b": M([[1, 1, 1], [0, 1, 2], [0, 0, 1]]),
    "18a": M([[1, 0], [0, 2]]),
    "18b": M([[0, 1], [0, 0]]),
    "18c": M([[1, 1], [0, 1]]),
    "18d": M([[0, 1, 0], [0, 0, 2], [0, 0, 0]]),
    "18e": M([[0, -1], [1, 0]]),
    "19f1": M([[A, 0, B, 0], [0, A, 0, B], [C, 0, D, 0], [0, C, 0, D]]),
    "19f2": M([[A, C, 0, 0], [B, D, 0, 0], [0, 0, A, C], [0, 0, B, D]]),
    "19f3": M([[0, -C, B, 0], [-B, A - D, 0, B], [C, 0, D - A, -C], [0, C, -B, 0]]),
    "22": M([[1, 1, 0, 0], [0, 1, 2, 0], [0, 0, 1, 3], [0, 0, 0, 1]]),
    "22inverse": M([[1, -1, 2, -6], [0, 1, -2, 6], [0, 0, 1, -3], [0, 0, 0, 1]]),
    "28f1": M([[1, 0], [0, 3], [-1, 1], [1, 1]]),
    "28f2": M([[1, 1], [0, 0], [0, 0], [0, 0]]),
    "28f3": M([[1, 1, 0, 0], [0, 0, 1, 1]]),
    "28f4": M([[0, 0, 0, 0], [0, 0, 1, 0]]),
    "28somme": M([[2, 1], [0, 3], [-1, 1], [1, 1]]),
    "28double": M([[2, 2], [0, 0], [0, 0], [0, 0]]),
    "28difference": M([[-1, -1, 0, 0], [0, 0, 2, -1]]),
    "28f4f1": M([[0, 0], [-1, 1]]),
    "28f2f3": M([[1, 1, 1, 1], [0] * 4, [0] * 4, [0] * 4]),
    "28f3f2f4": M([[0, 0, 1, 0], [0, 0, 0, 0]]),
}
"""Cle : la matrice que le syllabus imprime."""


@pytest.mark.parametrize("cle", sorted(ORACLES))
def test_exercices_14_a_19_22_et_28(cle: str) -> None:
    calculee = matrices.toutes()[cle]
    assert sp.simplify(calculee - ORACLES[cle]).is_zero_matrix


@pytest.mark.parametrize("ordre", [2, 3, 5])
def test_exercice_15b_l_homothetie(ordre: int) -> None:
    assert matrices.homothetie(ordre) == 7 * sp.eye(ordre)


def test_exercice_22_la_primitive_se_lit_dans_l_inverse() -> None:
    t = matrices.T
    primitive = matrices.primitive_du_22()
    assert sp.simplify(sp.diff(primitive, t) - t**3 * sp.exp(t)) == 0
    attendue = (-6 + 6 * t - 3 * t**2 + t**3) * sp.exp(t)
    assert sp.simplify(primitive - attendue) == 0


TIERS = R(1, 3)
PASSAGES = {
    "17": M([[1, 2, 2], [0, -3, -2]]),
    "20Pvv": TIERS * M([[2, 0, 1], [-3, 0, 3], [1, 3, 2]]),
    "20Pvprimev": TIERS * M([[3, -1, 0], [-3, -1, 3], [3, 2, 0]]),
    "21a": M(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, R(1, 2), 0, 0],
            [0, 0, TIERS, 0],
            [0, 0, 0, R(1, 4)],
        ]
    ),
    "21Pve": M([[1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
    "21c": M(
        [
            [-1, -1, -1, 0],
            [1, R(1, 2), R(5, 6), R(1, 12)],
            [0, R(1, 2), R(1, 6), -R(1, 12)],
            [0, 0, TIERS, R(1, 12)],
            [0, 0, 0, R(1, 4)],
        ]
    ),
    "21Pvprimee": M(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    ),
    "23": M(
        [
            [I / 2, -R(1, 2), -I / 2, R(1, 2)],
            [0, 0, I, 0],
            [-I / 2, R(1, 2), I / 2, R(1, 2)],
            [I / 2, R(1, 2), -I / 2, -R(1, 2)],
        ]
    ),
    "24inverse": M([[-2, 5, -7], [-2, 4, -5], [1, -2, 3]]),
    "25a": M([[0, 1, 0], [0, 0, 2]]),
    "25Pv": M([[5, 0, 3], [-2, -3, 1], [1, 2, 0]]),
    "25Pw": M([[-2, -3], [2, 4]]),
    "25c": M([[1, 0, -2], [0, 1, 1]]),
    "26vv": sp.diag(1, 1, -3),
    "26ee": M([[1, 0, 0], [-8, -TIERS, R(8, 3)], [8, R(4, 3), -R(5, 3)]]),
    "27": M([[1, -1, 1, 1], [1, 0, 2, -1], [1, 1, 3, -3]]),
    "29Pve": M([[1, 0, 0], [0, 1, 0], [1, 1, 1]]),
    "29Pev": M([[1, 0, 0], [0, 1, 0], [-1, -1, 1]]),
    "29fvv": sp.diag(1, 1, 0),
    "29gvv": sp.diag(0, 0, 1),
    "29fee": M([[1, 0, 0], [0, 1, 0], [1, 1, 0]]),
    "29gee": M([[0, 0, 0], [0, 0, 0], [-1, -1, 1]]),
    "30a": M([[-12, -41, -8], [8, 24, 5]]),
    "32": M([[-1, 0, 0, 1], [-1, 1, 1, 0], [-1, 1, 1, 0], [-1, 0, 0, 1]]),
}


@pytest.mark.parametrize("cle", sorted(PASSAGES))
def test_exercices_17_a_32_matrices_et_changements_de_base(cle: str) -> None:
    calculee = bases.toutes()[cle]
    assert sp.simplify(calculee - PASSAGES[cle]).is_zero_matrix


def test_exercice_20_les_coordonnees_dans_les_deux_bases() -> None:
    dans_v, dans_v_prime = bases.coordonnees_du_20()
    assert dans_v == (TIERS, -R(2, 3), TIERS)
    assert dans_v_prime == (TIERS, 0, -TIERS)


def test_exercice_27_noyau_rang_et_supplementaire() -> None:
    noyau, rang = suite.noyau_et_rang_du_27()
    assert rang == 2
    assert set(noyau) == {(-2, -1, 1, 0), (1, 2, 0, 1)}
    assert sont_supplementaires(noyau, ((0, 0, 1, 0), (0, 0, 0, 1)))


def test_exercice_30b_deux_matrices_qui_reduisent() -> None:
    """Celles du syllabus conviennent, et celles que la bibliotheque trouve aussi."""
    reduite = M([[1, 0, 0], [0, 1, 0]])
    p_syllabus = M([[1, 0, -23], [0, 1, 17], [0, 0, 13]])
    q_syllabus = M([[4, 5], [1, -2]]) / 13
    assert q_syllabus * suite.A_30 * p_syllabus == reduite
    q, p = suite.reduction_du_30()
    assert q.det() != 0
    assert p.det() != 0
    assert q * suite.A_30 * p == reduite


def test_exercice_31_l_application_retrouvee() -> None:
    assert suite.exercice_31() == (3 * X - 2 * Y + Z, X + 3 * Y, X - Y + Z)


def test_une_cle_devient_un_nom_de_macro_sans_chiffre() -> None:
    assert nom_de_la_cle("AlgHuit", "28f3f2f4") == "AlgHuitVingtHuitFTroisfDeuxfQuatre"
    assert nom_de_la_cle("AlgHuit", "20Pvv") == "AlgHuitVingtPvv"
    assert nom_de_la_cle("AlgSept", "36W1", "Base") == "AlgSeptTrenteSixWUnBase"


def test_une_cle_sans_numero_est_refusee() -> None:
    with pytest.raises(ValueError, match="cle sans numero d'exercice : "):
        nom_de_la_cle("AlgHuit", "Pvv")


def test_le_jeu_de_macros_de_la_partie_b() -> None:
    macros = macros_de_la_partie_b()
    assert all(nom.startswith("AlgHuit") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgHuitQuatorzeA"] == r"\left(5,\, 3\right)"
    assert macros["AlgHuitVingtB"] != macros["AlgHuitVingtC"]
    assert macros["AlgHuitVingtC"].endswith(r"- \frac{1}{3}\right)")
    assert macros["AlgHuitTrenteUn"].startswith(r"\left(3 x - 2 y + z")
    assert r"\frac{1}{12}" in macros["AlgHuitVingtUnC"]
    assert "t^{3}" in macros["AlgHuitVingtDeuxPrimitive"]
    assert "algebre-08b-matrices-associees" in JEUX_DE_MACROS
