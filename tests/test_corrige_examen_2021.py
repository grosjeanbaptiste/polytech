"""Le corrige de l'examen de janvier 2021 : ce que la bibliotheque lui fournit.

Source : `Mathematique/EXAMENS/Exam_2021_Janvier_Enonce.pdf`. Les reponses
elles-memes sont recoupees par `tests/test_examen_de_mathematique_2021.py` ;
ce fichier tient la couche d'application -- les expressions que le corrige
compose -- et le jeu de macros publie.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import examen_2021_analyse as an
from polytech.maths.application import examen_2021_systemes as sy
from polytech.maths.application.macros_examen_2021 import macros_de_l_examen_2021
from polytech.maths.domain.algebre.solutions import PARAMETRES, Nature
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as T
from polytech.maths.domain.symboles import VARIABLE as X

ALPHA, BETA = PARAMETRES[:2]
A = sy.PARAMETRE


def _derive_en(primitive: sp.Expr, integrande: sp.Expr) -> bool:
    return bool(sp.simplify(sp.diff(primitive, X) - integrande) == 0)


def test_les_primitives_i_et_j_de_la_question_1() -> None:
    """La primitive `K` est tenue par `tests/test_courbes_des_examens.py`."""
    assert _derive_en(an.primitive_i(), X / (1 + sp.cos(X)))
    positif = sp.Symbol("x", positive=True)
    j = an.primitive_j().subs(X, positif)
    fraction = (1 + 2 * positif - 3 * positif**2) / (positif - positif**3)
    assert sp.simplify(sp.diff(j, positif) - fraction) == 0


def test_l_etape_par_parties_de_i_laisse_la_tangente_a_integrer() -> None:
    etape = an.etape_i()
    assert etape.atoms(sp.Integral)
    assert sp.simplify(etape.doit() - an.primitive_i()).diff(X).simplify() == 0


def test_les_fractions_simples_de_j() -> None:
    assert sp.simplify(sum(an.termes_j()) - (1 / X + 2 / (X + 1))) == 0


def test_le_systeme_4_se_discute_par_le_rang() -> None:
    assert sy.discussion_4().generique is Nature.INDETERMINE
    assert sy.discussion_4().particulieres[sp.Integer(-1)] is Nature.IMPOSSIBLE
    assert sy.solution_4_en(sp.Integer(1)) == (5 - 2 * ALPHA, ALPHA, BETA)


def test_la_solution_generique_du_systeme_4_verifie_ses_trois_equations() -> None:
    x, y, z = sy.solution_4_generique()
    lignes = sy.SYSTEME_4.matrice * sp.Matrix([x, y, z]) - sy.SYSTEME_4.second_membre
    assert sp.simplify(lignes) == sp.zeros(3, 1)
    assert ALPHA in sp.Matrix([x, y, z]).free_symbols


def test_le_systeme_5_et_ses_trois_valeurs_singulieres() -> None:
    assert sp.expand(sy.determinant_5() + A**2 * (A - 1) ** 3 * (A + 1)) == 0
    assert sy.solution_5_en(sp.Integer(-1)) == (-ALPHA, 0, ALPHA)
    assert sy.solution_5_en(sp.Integer(0)) == (-ALPHA - BETA, ALPHA, BETA)
    assert sy.solution_5_en(sp.Integer(1)) == (-ALPHA - BETA, ALPHA, BETA)
    assert sy.solution_5_en(sp.Integer(2)) == (0, 0, 0)


def test_les_trois_equations_de_la_question_3() -> None:
    attendues = (
        sp.exp(T**2) - sp.Rational(1, 2),
        8 * sp.cos(T) - 2 * sp.cos(T) ** 2,
        (1 - 5 * sp.exp(sp.cos(T))) / sp.sin(T),
    )
    for numero, attendue in zip((7, 8, 9), attendues, strict=True):
        assert sp.simplify(an.solution_de(numero) - attendue) == 0


def test_une_equation_inconnue_est_refusee() -> None:
    with pytest.raises(KeyError, match="pas d'equation 10 a la question 3"):
        an.solution_de(10)


def test_les_trois_valeurs_de_i_deux() -> None:
    dix, onze, douze = an.i_deux()
    assert sp.simplify(dix - (sp.pi / 8 + sp.Rational(1, 4))) == 0
    positif = sp.Symbol("x", positive=True)
    assert (
        sp.simplify(sp.diff(onze.subs(X, positif), positif) - sp.log(positif) ** 2) == 0
    )
    assert sp.simplify(douze - (sp.sqrt(2) - sp.log(1 + sp.sqrt(2))) / 2) == 0


def test_le_jeu_de_macros_porte_chaque_reponse() -> None:
    macros = macros_de_l_examen_2021()
    assert all(nom.startswith("ExamVingtEtUn") for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    for attendu in (
        "PrimitiveI",
        "PrimitiveJ",
        "PrimitiveK",
        "SolutionSept",
        "IDeuxDix",
    ):
        assert f"ExamVingtEtUn{attendu}" in macros
