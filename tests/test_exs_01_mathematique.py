"""Mathematique, exercices 1 : integration, matrices et determinants.

Oracles : les solutions imprimees dans la feuille elle-meme
(`Mathematique/EXS/Exs_01_Integration_matrices_determinants.pdf`, pages 3
a 6, lues sur les pages rendues -- la couche de texte perd les signes).

Chaque reponse de la bibliotheque est controlee par deux chemins qui ne
lui doivent rien : la *derivation* (une primitive derivee rend son
integrande, a trente chiffres, en des points du domaine) et la
*quadrature* numerique de `mpmath` pour les integrales definies. La
comparaison a la reponse imprimee vient en plus : deux primitives justes
ne different que d'une constante.

Aucun ecart : les quarante-huit reponses de la feuille sont justes. Le
corrige leur ajoute le chemin, et une primitive de l'exercice 12 valable
sur tout R, que la feuille reclame sans la donner.
"""

from __future__ import annotations

import itertools
import math

import mpmath as mp
import pytest
import sympy as sp

from polytech.maths.application import algebre_exs_01 as al
from polytech.maths.application import courbes_exs_01 as co
from polytech.maths.application import definies_exs_01 as de
from polytech.maths.application import newton_exs_01 as nw
from polytech.maths.application import primitives_exs_01 as pr
from polytech.maths.domain.symboles import VARIABLE as X

A, B = pr.A, pr.B
VALEURS = {A: sp.Rational(17, 10), B: sp.Rational(3, 5)}
POINTS = (sp.Rational(31, 100), sp.Rational(57, 100))
"""Deux points de ]0, 1[, ou toutes les fonctions de la feuille sont definies."""

OFFICIELLES = {
    "1": sp.exp(sp.sin(X) ** 2),
    "2a": (15 * sp.sin(X) - 10 * sp.sin(X) ** 3 + 3 * sp.sin(X) ** 5) / 15,
    "2b": sp.sinh(X) + sp.sinh(X) ** 3 / 3,
    "2c": (sp.sin(4 * X) + 8 * sp.sin(2 * X) + 12 * X) / 32,
    "2d": (sp.sinh(4 * X) - 8 * sp.sinh(2 * X) + 12 * X) / 32,
    "3": (X**3 - 3 * X**2 + 6 * X - 6) * sp.exp(X),
    "4a": X * sp.log(X) - X,
    "4b": X**2 / 2 * sp.log(X) - X**2 / 4,
    "4c": X * sp.asin(X) + sp.sqrt(1 - X**2),
    "5": (sp.sinh(X) * sp.sin(X) - sp.cosh(X) * sp.cos(X)) / 2,
    "6": sp.log(sp.Abs(sp.tan(X / 2))),
    "7": A**2 / 2 * sp.asin(X / A) + X / 2 * sp.sqrt(A**2 - X**2),
    "8": sp.Rational(2, 3) * sp.sqrt(sp.exp(X) + 1) * (sp.exp(X) - 2),
    "9a": sp.exp(A * X) * (A * sp.cos(B * X) + B * sp.sin(B * X)) / (A**2 + B**2),
    "9b": sp.exp(A * X) * (A * sp.sin(B * X) - B * sp.cos(B * X)) / (A**2 + B**2),
    "10": 2 * sp.sqrt(X / (1 - X)) - 2 * sp.atan(sp.sqrt(X / (1 - X))),
    "11": (sp.asin(X) - X * sp.sqrt(1 - X**2)) / 2,
    "12": sp.atan(sp.tan(X / 2) + 1),
    "13": sp.Rational(2, 3) * sp.asin(sp.sqrt(X**3 / A**3)),
    "14": X / 2 - sp.exp(-2 * X) / 4,
}
"""Les reponses de la page 3, la variable de la feuille ecrite `x` partout."""


def _valeur(expression: sp.Expr, point: sp.Expr) -> sp.Expr:
    return sp.N(expression.subs(VALEURS).subs(X, point), 30)


def test_la_feuille_est_resolue_en_entier() -> None:
    """Vingt primitives, dans l'ordre de la feuille."""
    assert list(pr.PRIMITIVES) == list(OFFICIELLES)


@pytest.mark.parametrize("numero", list(OFFICIELLES))
def test_chaque_primitive_derivee_rend_son_integrande(numero: str) -> None:
    """Le seul controle qui ne depende d'aucune table : deriver la reponse."""
    resolution = pr.PRIMITIVES[numero]()
    ecart = sp.diff(resolution.primitive, X) - resolution.integrande
    for point in POINTS:
        assert abs(_valeur(ecart, point)) < sp.Float("1e-20")


@pytest.mark.parametrize("numero", list(OFFICIELLES))
def test_chaque_primitive_est_celle_de_la_feuille_a_une_constante_pres(
    numero: str,
) -> None:
    """La difference avec la reponse imprimee vaut la meme chose en deux points."""
    difference = pr.PRIMITIVES[numero]().primitive - OFFICIELLES[numero]
    premier, second = (_valeur(difference, point) for point in POINTS)
    assert abs(premier - second) < sp.Float("1e-20")


@pytest.mark.parametrize("numero", list(OFFICIELLES))
def test_la_reponse_imprimee_est_elle_meme_une_primitive(numero: str) -> None:
    """La feuille n'est pas crue sur parole : sa reponse aussi est derivee."""
    ecart = sp.diff(OFFICIELLES[numero], X) - pr.PRIMITIVES[numero]().integrande
    for point in POINTS:
        assert abs(_valeur(ecart, point)) < sp.Float("1e-20")


def test_ex2_les_formes_linearisees() -> None:
    """Page 3 : `sin 5t / 80 + 5 sin 3t / 48 + 5 sin t / 8`, et son analogue."""
    cosinus, hyperbolique = pr.formes_linearisees()
    attendue = sp.sin(5 * X) / 80 + sp.Rational(5, 48) * sp.sin(3 * X)
    attendue += sp.Rational(5, 8) * sp.sin(X)
    assert sp.simplify(cosinus - attendue) == 0
    seconde = sp.sinh(3 * X) / 12 + sp.Rational(3, 4) * sp.sinh(X)
    assert (
        sp.simplify(sp.expand(hyperbolique.rewrite(sp.exp) - seconde.rewrite(sp.exp)))
        == 0
    )


def test_ex3_trois_integrations_par_parties() -> None:
    """Chaque geste abaisse le degre de un : trois gestes pour `x^3`."""
    etapes = pr.PRIMITIVES["3"]().etapes
    assert len(etapes) == 3
    assert etapes[0] == X**3 * sp.exp(X) - sp.Integral(3 * X**2 * sp.exp(X), X)


def test_ex5_et_ex9_l_integrale_revient_et_se_resout() -> None:
    """Deux parties ramenent l'integrale cherchee : une equation en `I`."""
    for numero in ("5", "9a", "9b"):
        resolution = pr.PRIMITIVES[numero]()
        assert resolution.methode == "deux integrations par parties, puis une equation"
        assert len(resolution.etapes) == 2


def test_ex12_une_primitive_sur_tout_r() -> None:
    """La feuille : « ne cherchait-on pas une primitive sur R ? ».

    `x/2 + arctan((1 + cos x) / (2 + sin x))` est derivable partout, et
    coincide avec `arctan(tan(x/2) + 1)` sur ]-pi, pi[.
    """
    partout = pr.primitive_du_12_sur_r()
    integrande = pr.PRIMITIVES["12"]().integrande
    for point in (
        sp.Rational(1, 3),
        sp.Integer(3),
        sp.pi,
        sp.Integer(4),
        sp.Integer(-5),
    ):
        ecart = sp.diff(partout, X).subs(X, point) - integrande.subs(X, point)
        assert abs(sp.N(ecart, 30)) < sp.Float("1e-20")
    for point in POINTS:
        assert _valeur(partout - OFFICIELLES["12"], point) == pytest.approx(
            0, abs=1e-25
        )
    saut = sp.limit(OFFICIELLES["12"], X, sp.pi, "-") - sp.limit(
        OFFICIELLES["12"], X, sp.pi, "+"
    )
    assert saut == sp.pi


DEFINIES = {
    1: 3 * sp.pi / 16,
    2: sp.Rational(4, 15),
    3: sp.pi**2 / 4 - 2,
    4: sp.Integer(0),
    5: sp.pi / 4,
    6: sp.Integer(1),
    7: -sp.log(sp.sqrt(2) - 1) / sp.sqrt(2),
    8: 4 * (2 - (A + 2) * sp.sqrt(1 - A)) / (3 * A**2),
    9: -sp.Rational(1, 4),
    10: sp.pi / 2 - 1,
    11: sp.log(sp.Rational(5, 2)) / 2,
    12: sp.pi / 4 - sp.pi**2 / 32 - sp.log(sp.sqrt(2)),
    13: 2 - sp.pi / 2,
    14: 2 + 2 * sp.log(2),
    15: 4 * sp.sqrt(2)
    - 2 * sp.sqrt(sp.E + 1)
    + 4 * sp.log((sp.sqrt(sp.E + 1) + 1) / (sp.sqrt(2) + 1))
    - 2,
    16: A * sp.log((1 - A) / (1 + A)) - sp.log(1 - A**2),
    17: sp.pi / 6 * (3 - 4 / sp.sqrt(3)),
    18: sp.log(1 + sp.sqrt(2)) + sp.sqrt(2),
    19: sp.log(1 + sp.sqrt(2)) + sp.sqrt(2),
}
"""Les dix-neuf valeurs de la page 4 ; le parametre `a` y est pris dans ]0, 1[."""
PARAMETRE = {A: sp.Rational(2, 5)}


def test_les_dix_neuf_integrales_sont_la() -> None:
    assert list(de.INTEGRALES) == list(DEFINIES)


@pytest.mark.parametrize("numero", list(DEFINIES))
def test_chaque_valeur_est_celle_de_la_feuille(numero: int) -> None:
    """La valeur calculee et la valeur imprimee, a trente chiffres."""
    calculee = de.INTEGRALES[numero]().valeur
    ecart = sp.N((calculee - DEFINIES[numero]).subs(PARAMETRE), 30)
    assert abs(ecart) < sp.Float("1e-20")


@pytest.mark.parametrize("numero", list(DEFINIES))
def test_chaque_valeur_est_celle_de_la_quadrature(numero: int) -> None:
    """Un chemin qui ne doit rien au calcul formel : l'integration numerique."""
    integrale = de.INTEGRALES[numero]()
    fonction = sp.lambdify(X, integrale.integrande.subs(PARAMETRE), "mpmath")
    bornes = [float(sp.N(b)) for b in integrale.bornes]
    numerique = mp.quad(fonction, bornes)
    attendue = float(sp.N(integrale.valeur.subs(PARAMETRE)))
    assert float(numerique.real) == pytest.approx(attendue, rel=1e-9, abs=1e-12)


def test_les_integrales_4_18_et_19_par_la_parite() -> None:
    """Une impaire sur un intervalle symetrique est nulle ; 18 se ramene a 19."""
    assert de.INTEGRALES[4]().methode.startswith("integrande impair")
    assert de.INTEGRALES[18]().valeur == de.INTEGRALES[19]().valeur
    conjuguee = sp.sqrt(X**2 + 1) - X
    assert sp.simplify(de.INTEGRALES[18]().integrande - conjuguee) == 0


def test_l_integrale_4_porte_ses_bornes_symetriques() -> None:
    assert de.INTEGRALES[4]().bornes == (-sp.pi / 2, sp.pi / 2)


def test_un_integrande_pair_n_est_pas_annule_par_la_parite() -> None:
    with pytest.raises(ValueError, match="non impair"):
        nw.impaire(X**2, sp.Integer(1))


def test_une_borne_se_prend_a_la_limite_depuis_l_interieur() -> None:
    """`arctan(1/x)` saute de `pi` en 0 : seule la limite a droite est la bonne.

    Sa derivee est `-1/(1 + x^2)`, dont l'integrale de 0 a 1 vaut `-pi/4`.
    Le meme saut, place a la borne haute, eprouve la limite a gauche.
    """
    integrande = -1 / (1 + X**2)
    basse = nw.newton_leibniz(integrande, (0, 1), sp.atan(1 / X), "essai")
    assert basse.valeur == -sp.pi / 4
    decalee = -1 / (1 + (X - 1) ** 2)
    haute = nw.newton_leibniz(decalee, (0, 1), sp.atan(1 / (X - 1)), "essai")
    assert haute.valeur == -sp.pi / 4


# --- Matrices et determinants -----------------------------------------


def test_ex21_le_determinant_et_l_inverse() -> None:
    """Page 5 : `det A = -1 - a^3`, singuliere en `a = -1` seulement."""
    a = al.PARAMETRE
    assert sp.expand(al.determinant()) == -1 - a**3
    assert al.valeurs_singulieres() == (-1,)
    comatrice = sp.Matrix([[-1, a, -(a**2)], [a, -(a**2), -1], [-(a**2), -1, a]])
    assert al.comatrice() == comatrice
    produit = sp.simplify(al.matrice() * al.inverse())
    assert produit == sp.eye(3)
    assert sp.simplify(al.inverse() - comatrice.T / (-1 - a**3)) == sp.zeros(3, 3)


def test_ex22_les_trois_systemes() -> None:
    """Page 6 : (0, 0, 0), (3, 2, 0), et la solution en `a`, `b`, `c`."""
    premier, deuxieme, troisieme = al.solutions_des_systemes()
    assert premier == (0, 0, 0)
    assert al.determinants_des_systemes() == (-7, -1, 18)
    assert deuxieme == (3, 2, 0)
    a, b, c = al.SECONDS_MEMBRES
    attendue = (
        (8 * a + 5 * b - c) / 18,
        (-2 * a + b + 7 * c) / 18,
        (-4 * a - 7 * b + 5 * c) / 18,
    )
    assert [sp.simplify(x - y) for x, y in zip(troisieme, attendue, strict=True)] == [
        0,
        0,
        0,
    ]


def test_ex23_le_premier_systeme_parametrique() -> None:
    """Page 6 : des solutions non nulles si et seulement si deux parametres egaux."""
    a, b, c = al.SECONDS_MEMBRES
    assert sp.factor(al.determinant_parametrique(1)) == sp.factor(
        (a - b) * (a - c) * (b - c)
    )
    assert al.solutions_non_nulles(1, {b: a}) == ((1, -1, 0),)
    assert al.solutions_non_nulles(1, {c: b}) == ((0, 1, -1),)
    assert al.solutions_non_nulles(1, {c: a}) == ((1, 0, -1),)
    plan = al.solutions_non_nulles(1, {b: a, c: a})
    assert len(plan) == 2
    assert all(sum(vecteur) == 0 for vecteur in plan)


def test_ex23_le_second_systeme_parametrique() -> None:
    """Page 6 : `2abc + bc + ca + ab = 1` ; alors `x` va comme `a / (1 + a)`."""
    a, b, c = al.SECONDS_MEMBRES
    condition = 2 * a * b * c + b * c + c * a + a * b - 1
    assert sp.expand(al.determinant_parametrique(2) + condition) == 0
    valeurs = {a: sp.Rational(1, 2), b: sp.Rational(1, 3)}
    c_solution = sp.solve(condition.subs(valeurs), c)[0]
    valeurs[c] = c_solution
    (vecteur,) = al.solutions_non_nulles(2, valeurs)
    attendu = [v / (1 + v) for v in (valeurs[a], valeurs[b], valeurs[c])]
    rapport = [sp.simplify(x / y) for x, y in zip(vecteur, attendu, strict=True)]
    assert rapport[0] == rapport[1] == rapport[2]
    assert sp.simplify(sum(attendu) - 1) == 0


def test_ex12_les_courbes_de_la_figure() -> None:
    """Trois branches qui sautent de `pi`, et la primitive continue qui les relie."""
    branches, (abscisses, continue_) = co.courbes_du_12()
    assert len(branches) == 3
    centrale_x, centrale_y = branches[1]
    assert centrale_x[0] > -math.pi
    assert centrale_x[-1] < math.pi
    assert centrale_y[0] == pytest.approx(-math.pi / 2, abs=0.05)
    assert centrale_y[-1] == pytest.approx(math.pi / 2, abs=0.05)
    assert (abscisses[0], abscisses[-1]) == pytest.approx((-3 * math.pi, 3 * math.pi))
    milieu = len(centrale_x) // 2
    attendue = float(pr.primitive_du_12_sur_r().subs(X, centrale_x[milieu]))
    assert centrale_y[milieu] == pytest.approx(attendue)
    for gauche, droite in itertools.pairwise(branches):
        assert droite[1][0] - gauche[1][-1] == pytest.approx(-math.pi, abs=0.1)
    pentes = [b - a for a, b in itertools.pairwise(continue_)]
    assert min(pentes) > 0
