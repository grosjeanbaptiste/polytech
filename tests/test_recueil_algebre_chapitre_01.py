"""Le chapitre I du recueil d'algebre, « Les nombres complexes », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 18 des
sections I.1.4, I.2.5, I.3.2, I.4.3 et I.5.3 ; l'oracle est la page
« Solutions du 1er chapitre » (pages 14 a 16 du syllabus). Les exercices de
raisonnement -- 7, 9, 10, 16 et 17 -- n'ont pas de formule a publier : ils
sont verifies ici, directement.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch01_calculs as calculs
from polytech.maths.application import recueil_algebre_ch01_lieux as lieux
from polytech.maths.application import recueil_algebre_ch01_racines as racines
from polytech.maths.domain.algebre.complexes_exacts import (
    ORDONNEE as Y,
)
from polytech.maths.domain.algebre.complexes_exacts import (
    forme_polaire,
    racines_de_l_unite,
)
from polytech.maths.domain.symboles import VARIABLE as X

IMAG = sp.I
R3 = sp.sqrt(3)
DEMI = sp.Rational(1, 2)


def _egaux(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(sp.simplify(sp.expand_complex(a - b)) == 0)


def _memes(calcules: tuple[sp.Expr, ...], attendus: tuple[sp.Expr, ...]) -> bool:
    return len(calcules) == len(attendus) and all(
        any(_egaux(c, a) for c in calcules) for a in attendus
    )


def _meme_angle(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(sp.simplify((a - b) / (2 * sp.pi)).is_integer)


@pytest.mark.parametrize(
    ("lettre", "attendu"),
    [
        ("a", sp.Integer(3)),
        ("b", 71 - 65 * IMAG),
        ("c", 7 + 24 * IMAG),
        ("d", sp.Rational(18, 25) - IMAG / 25),
        ("e", IMAG),
    ],
)
def test_exercice_1_parties_reelle_et_imaginaire(lettre: str, attendu: sp.Expr) -> None:
    assert _egaux(calculs.exercice_1()[lettre], attendu)


@pytest.mark.parametrize(
    ("lettre", "module", "angle"),
    [
        ("a", 2, sp.pi / 3),
        ("b", 2, sp.pi / 2),
        ("c", 3, sp.pi),
        ("d", 1 / sp.sqrt(2), -sp.pi / 4),
        ("e", sp.sqrt(41), sp.pi + sp.atan(-sp.Rational(4, 5))),
    ],
)
def test_exercice_3_module_et_argument(
    lettre: str, module: sp.Expr, angle: sp.Expr
) -> None:
    calcule = calculs.exercice_3()[lettre]
    assert _egaux(calcule[0], sp.sympify(module))
    assert _meme_angle(calcule[1], angle)


def test_exercice_4_un_quotient_de_puissances() -> None:
    module, angle = calculs.exercice_4()
    assert _egaux(module, 2 ** sp.Rational(-5, 2))
    assert _meme_angle(angle, -7 * sp.pi / 12)


def test_exercices_5_et_6_par_moivre() -> None:
    assert _egaux(calculs.exercice_5(), -8 + 8 * R3 * IMAG)
    assert _egaux(calculs.exercice_6(), sp.Integer(-64))


def test_exercice_7_le_produit_ajoute_les_arguments() -> None:
    phi = sp.Symbol("varphi", real=True)
    gauche = (1 + IMAG * R3) * (1 + IMAG) * (sp.cos(phi) + IMAG * sp.sin(phi))
    droite = 2 * sp.sqrt(2) * sp.exp(IMAG * (7 * sp.pi / 12 + phi))
    assert abs(sp.N((gauche - droite).subs(phi, sp.Rational(3, 10)), 30)) < 1e-25
    assert forme_polaire((1 + IMAG * R3) * (1 + IMAG)) == (
        2 * sp.sqrt(2),
        7 * sp.pi / 12,
    )


def test_exercice_8_produit_et_quotient() -> None:
    reponses = calculs.exercice_8()
    assert _egaux(reponses["produit"][0], sp.Integer(6))
    assert _meme_angle(reponses["produit"][1], 4 * sp.pi / 3)
    assert _egaux(reponses["quotient"][0], sp.Rational(2, 3))
    assert _meme_angle(reponses["quotient"][1], -5 * sp.pi / 3)


def test_exercice_9_trois_equations() -> None:
    """a) `y = 0`, `x >= 0` ; b) `y = 0` ; c) `x = 0`, `y = pi + 2 k pi`."""
    assert sp.Abs(3 + 0 * IMAG) == 3
    assert sp.Abs(-3 + 0 * IMAG) != -3
    assert _egaux(5 * sp.exp(IMAG * 0), sp.Integer(5))
    k = sp.Symbol("k", integer=True)
    assert sp.simplify(sp.exp(0 + IMAG * (-sp.pi + 2 * k * sp.pi))) == -1


@pytest.mark.parametrize("n", [3, 4, 5])
def test_exercice_10_le_conjugue_egale_une_puissance(n: int) -> None:
    """Pour `n > 2` : `z = 0` ou `z` racine `n`-ieme de l'unite."""
    for z in (sp.Integer(0), *racines_de_l_unite(n)):
        assert _egaux(sp.conjugate(z), z ** (n - 1))
    assert not _egaux(sp.conjugate(2 * IMAG), (2 * IMAG) ** (n - 1))


@pytest.mark.parametrize(
    ("lettre", "attendu"),
    [
        ("a", -2 + 3 * IMAG),
        ("b", -sp.Rational(5, 2) + IMAG * (R3 / 2 - 2)),
        ("c", 6 * sp.exp(IMAG * sp.pi / 30)),
    ],
)
def test_exercice_11_operations(lettre: str, attendu: sp.Expr) -> None:
    assert _egaux(calculs.exercice_11()[lettre], attendu)


def test_exercice_12_quatre_ensembles_du_plan_de_gauss() -> None:
    ensembles = lieux.exercice_12()
    assert sp.expand(ensembles["A"] - (X + Y - 3)) == 0
    assert sp.expand(ensembles["B"] - (X**2 - Y**2)) == 0
    assert sp.solve(ensembles["C"], X) == [0]
    assert sp.solve(ensembles["D"], Y) == [X]


def test_exercice_13_un_disque_et_un_demi_plan() -> None:
    ensembles = lieux.exercice_13()
    assert sp.expand(ensembles["A"] - (X**2 + Y**2 - 9)) == 0
    assert sp.solve(ensembles["B"] < 0, X) == sp.solve(sp.Rational(3, 2) > X, X)


def test_exercice_14_racines_de_l_unite() -> None:
    reponses = racines.exercice_14()
    assert _memes(reponses[3], (1, -DEMI + R3 / 2 * IMAG, -DEMI - R3 / 2 * IMAG))
    assert _memes(reponses[4], (1, IMAG, -1, -IMAG))
    assert len(reponses[6]) == 6
    assert all(_egaux(z**6, sp.Integer(1)) for z in reponses[6])


def test_exercice_15_racines_d_un_complexe() -> None:
    reponses = racines.exercice_15()
    r5 = sp.sqrt(5)
    assert _memes(reponses["a"], (r5 * (-1 + IMAG), r5 * (1 - IMAG)))
    assert any(_egaux(z, 1 + IMAG) for z in reponses["b"])
    assert all(_egaux(z**3, -2 + 2 * IMAG) for z in reponses["b"])
    assert _memes(reponses["c"], (3, 3 * IMAG, -3, -3 * IMAG))
    assert all(_egaux(z**5, sp.Integer(-1)) for z in reponses["d"])


def test_exercice_16_des_conjugues_donnent_les_racines_cinquiemes() -> None:
    for z in racines_de_l_unite(5):
        assert _egaux(z**7, sp.conjugate(1 / z**2))
    assert not _egaux(IMAG**7, sp.conjugate(1 / IMAG**2))


@pytest.mark.parametrize("n", [2, 3, 6])
def test_exercice_17_sommes_sur_les_racines_de_l_unite(n: int) -> None:
    etas = racines_de_l_unite(n)
    assert _egaux(sum(etas), sp.Integer(0))
    assert _egaux(sum(e**n for e in etas), sp.Integer(n))
    assert _egaux(sum(e ** (n + 1) for e in etas), sp.Integer(0))
    epsilon = etas[1]
    ponderee = sum((k + 1) * epsilon**k for k in range(n))
    assert _egaux(ponderee, -n / (1 - epsilon))


def test_exercice_18_cinq_equations() -> None:
    reponses = racines.exercice_18()
    assert _memes(
        reponses["a"],
        (-3, R3 + IMAG, 2 * IMAG, -R3 + IMAG, -R3 - IMAG, -2 * IMAG, R3 - IMAG),
    )
    r31 = sp.sqrt(31)
    assert _memes(reponses["b"], ((5 + r31 * IMAG) / 4, (5 - r31 * IMAG) / 4))
    assert _memes(reponses["c"], (IMAG, sp.Rational(3, 2) + IMAG / 2))
    assert _memes(reponses["d"], (2 + IMAG, 2 - IMAG, -2 + IMAG, -2 - IMAG))
    assert _memes(reponses["e"], (2, IMAG, -3 * IMAG))


def test_exercice_2_une_puissance_quatrieme_reelle() -> None:
    """Reel : `xy (x^2 - y^2) = 0` ; sur les axes la valeur est `x^4` ou `y^4`."""
    reelle, imaginaire = lieux.exercice_2()
    assert sp.expand(imaginaire - 4 * X * Y * (X**2 - Y**2)) == 0
    assert reelle.subs(Y, 0) == X**4
    assert reelle.subs(X, 0) == Y**4
    assert sp.expand(reelle.subs(Y, X) + 4 * X**4) == 0


def test_un_polynome_dont_les_racines_echappent_est_refuse() -> None:
    """`x^5 - x + 1` n'a pas de racine exprimable : mieux vaut lever que se taire."""
    with pytest.raises(ArithmeticError, match="racines manquantes pour "):
        racines.racines_exactes(X**5 - X + 1)
