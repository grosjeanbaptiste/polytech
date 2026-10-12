"""Le chapitre VI du recueil d'algebre, « Le rang d'une matrice », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, les cinq matrices de
la section VI.4 ; l'oracle est « Solutions du 6eme chapitre » (page 70 du
syllabus). Chaque rang est etabli par deux chemins : l'echelonnement, et le
plus grand mineur non nul.
"""

from __future__ import annotations

import re

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch06 as chapitre
from polytech.maths.application.macros_recueil_algebre_ch06 import (
    macros_du_chapitre_six,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.rang_exact import rang_par_les_mineurs

R = sp.Rational


@pytest.mark.parametrize(("lettre", "attendu"), [("a", 2), ("b", 3)])
def test_les_deux_rangs_numeriques(lettre: str, attendu: int) -> None:
    assert chapitre.rangs_numeriques()[lettre] == attendu
    assert rang_par_les_mineurs(chapitre.NUMERIQUES[lettre]) == attendu


@pytest.mark.parametrize(
    ("lettre", "chutes"),
    [
        ("c", {sp.Integer(1): 1, sp.Integer(-2): 2}),
        ("d", {sp.Integer(-1): 1, sp.Integer(1): 2}),
        ("e", {sp.Integer(1): 2, R(3, 2): 2}),
    ],
)
def test_les_trois_rangs_a_parametre(lettre: str, chutes: dict) -> None:
    generique, particuliers = chapitre.rangs_a_parametre()[lettre]
    assert generique == 3
    assert particuliers == chutes


@pytest.mark.parametrize("lettre", ["c", "d", "e"])
def test_le_rang_ne_chute_qu_aux_racines_du_determinant(lettre: str) -> None:
    matrice, parametre = chapitre.PARAMETRIQUES[lettre]
    racines = set(sp.roots(chapitre.determinants()[lettre], parametre))
    assert racines == set(chapitre.rangs_a_parametre()[lettre][1])
    assert matrice.subs(parametre, 7).rank() == 3


def test_chaque_cas_singulier_a_sa_forme_echelonnee() -> None:
    formes = chapitre.formes_singulieres()
    assert set(formes) == {"cUn", "cMoinsDeux", "dMoinsUn", "dUn", "eUn", "eTroisDemis"}
    nulles = {
        cle: sum(1 for i in range(3) if f[i, :].is_zero_matrix)
        for cle, f in formes.items()
    }
    assert nulles == {
        "cUn": 2,
        "cMoinsDeux": 1,
        "dMoinsUn": 2,
        "dUn": 1,
        "eUn": 1,
        "eTroisDemis": 1,
    }


def test_le_jeu_de_macros_du_chapitre_six() -> None:
    macros = macros_du_chapitre_six()
    assert all(re.fullmatch(r"AlgSix[A-Za-z]+", nom) for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgSixARang"] == "2"
    assert macros["AlgSixBRang"] == "3"
    assert macros["AlgSixEDet"] == r"3 \left(t - 1\right) \left(2 t - 3\right)"
    assert "algebre-06-rang" in JEUX_DE_MACROS


def test_les_relations_que_le_corrige_lit_dans_les_matrices_chiffrees() -> None:
    """Le rang seul ne fixe pas l'enonce : ces relations le fixent mieux."""
    a, b = chapitre.NUMERIQUES["a"], chapitre.NUMERIQUES["b"]
    assert a.row(2) == a.row(0) + a.row(1)
    assert a.row(3) == a.row(1) - a.row(0)
    assert b.col(3) == 2 * b.col(0)
    assert b.row(3) == b.row(0) + b.row(2)
    attendue = sp.Matrix([[1, -3, 2, 2], [0, 3, 6, 0], [0, 0, -12, 0], [0, 0, 0, 0]])
    assert chapitre.echelonnees()["b"].matrice == attendue
