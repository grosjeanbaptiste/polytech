"""Le chapitre IV du recueil d'algebre, « Les polynomes », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 6 de
la section IV.4 ; l'oracle est « Solutions du 4eme chapitre » (page 54 du
syllabus).
"""

from __future__ import annotations

import re

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch04 as chapitre
from polytech.maths.application.macros_recueil_algebre_ch04 import (
    macros_du_chapitre_quatre,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.polynomes_exacts import ordre_de_la_racine
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
R2 = sp.sqrt(2)


@pytest.mark.parametrize(
    ("lettre", "quotient", "reste"),
    [("a", X**2 + X - 2, -1), ("b", 3 * X**2 + 2, -21 * X**2 + 5 * X - 9)],
)
def test_exercice_1_divisions(lettre: str, quotient: sp.Expr, reste: sp.Expr) -> None:
    calcule = chapitre.exercice_1()[lettre]
    assert sp.expand(calcule[0] - quotient) == 0
    assert sp.expand(calcule[1] - reste) == 0
    dividende, diviseur = chapitre.DIVISIONS[lettre]
    assert sp.expand(diviseur * calcule[0] + calcule[1] - dividende) == 0


@pytest.mark.parametrize(
    ("lettre", "attendues"),
    [
        ("a", {-2, -1, 1, 2}),
        ("b", {R(7, 3), 2 + R2, 2 - R2}),
        ("c", {-2, 3, -R(1, 2), R(1, 3)}),
    ],
)
def test_exercice_2_racines(lettre: str, attendues: set) -> None:
    racines = chapitre.exercice_2()[lettre]
    assert {sp.nsimplify(r) for r in racines} == {sp.sympify(a) for a in attendues}
    assert all(chapitre.POLYNOMES_2[lettre].subs(X, r).equals(0) for r in racines)


@pytest.mark.parametrize(
    ("lettre", "attendue"),
    [
        ("a", (X**2 + 3) * (X**2 + 1)),
        ("b", (X - 3) * (X + 4) * (X**2 + 1)),
        ("c", (X - 1) * (X + 1) * (3 * X - 2) * (2 * X - 3)),
        ("d", (X**2 + X + 1) * (X**2 - X + 1)),
    ],
)
def test_exercice_3_factorisations_reelles(lettre: str, attendue: sp.Expr) -> None:
    dominant, facteurs = chapitre.exercice_3()[lettre]
    assert sp.expand(dominant * sp.Mul(*facteurs) - attendue) == 0
    assert len(facteurs) == len(sp.Mul.make_args(attendue))
    assert all(sp.degree(f, X) in (1, 2) for f in facteurs)


def test_exercice_4_une_racine_double_donnee() -> None:
    """`i` double, donc `-i` double : `(x^2 + 1)^2` divise, il reste un second degre."""
    quotient, racines = chapitre.exercice_4()
    assert sp.expand(quotient - (X**2 + X - 2)) == 0
    assert racines == {sp.I: 2, -sp.I: 2, sp.Integer(1): 1, sp.Integer(-2): 1}
    assert ordre_de_la_racine(chapitre.POLYNOME_4, sp.I) == 2


@pytest.mark.parametrize("n", [2, 3, 7])
def test_exercice_5_divisible_par_un_carre(n: int) -> None:
    a, b = chapitre.exercice_5()
    degre = chapitre.DEGRE
    polynome = (a * X**degre + b * X + 1).subs(degre, n)
    assert sp.simplify(a - 1 / (degre - 1)) == 0
    assert sp.simplify(b + degre / (degre - 1)) == 0
    assert sp.rem(sp.expand(polynome), (X - 1) ** 2, X) == 0


def test_exercice_6_fonctions_symetriques_des_racines() -> None:
    assert chapitre.exercice_6() == {"a": -6, "b": 9, "c": R(3, 2)}


def test_exercice_6_recoupe_par_les_racines_elles_memes() -> None:
    racines = sp.Poly(chapitre.POLYNOME_6, X).nroots(n=30)
    carres = sum(r**2 for r in racines)
    inverses = sum(1 / r for r in racines)
    assert abs(carres + 6) < 1e-20
    assert abs(inverses - R(3, 2)) < 1e-20


def test_le_jeu_de_macros_du_chapitre_quatre() -> None:
    macros = macros_du_chapitre_quatre()
    assert all(re.fullmatch(r"AlgQuatre[A-Za-z]+", nom) for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgQuatreSixC"] == r"\frac{3}{2}"
    assert macros["AlgQuatreUnAQuotient"] == "x^{2} + x - 2"
    assert macros["AlgQuatreTroisA"].startswith(r"\left(x^{2} + 1\right)")
    assert macros["AlgQuatreTroisC"].startswith("6 ")
    points = re.findall(r"\((-?[\d.]+),(-?[\d.]+)\)", macros["AlgQuatreCourbeDeuxA"])
    assert (float(points[0][0]), float(points[-1][0])) == (-2.4, 2.4)
    assert "algebre-04-polynomes" in JEUX_DE_MACROS


def test_exercice_2_ce_qui_reste_apres_les_racines_rationnelles() -> None:
    """Le quotient est exact, et ses racines sont celles qui manquaient."""
    quotients = chapitre.quotients_du_2()
    assert sp.expand(quotients["b"] - 3 * (X**2 - 4 * X + 2)) == 0
    assert sp.expand(quotients["c"] - (2 * X + 1) * (3 * X - 1)) == 0
    for lettre, diviseur in chapitre.DIVISEURS_2.items():
        produit = sp.expand(diviseur * quotients[lettre])
        assert sp.expand(produit - chapitre.POLYNOMES_2[lettre]) == 0


def test_les_macros_ne_confondent_ni_reste_ni_coefficient() -> None:
    macros = macros_du_chapitre_quatre()
    assert macros["AlgQuatreUnAReste"] == "-1"
    assert macros["AlgQuatreUnBReste"] == "- 21 x^{2} + 5 x - 9"
    assert macros["AlgQuatreCinqA"] == r"\frac{1}{n - 1}"
    assert macros["AlgQuatreCinqB"] == r"- \frac{n}{n - 1}"
