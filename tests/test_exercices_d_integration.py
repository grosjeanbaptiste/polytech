"""Mathematique, exercices 1 : primitives et integrales definies, reponses officielles.

Source : `Mathematique/EXS/Exs_01_Integration_matrices_determinants.pdf`,
pages 2 a 4 -- quatorze exercices de primitives avec leurs solutions,
dix-neuf integrales definies dont la valeur est donnee.

Chaque reponse officielle est controlee par un chemin qui ne doit rien a
un integrateur : une primitive se **derive**, une integrale definie se
**mesure** par quadrature numerique. Les deux gestes que les solutions
nomment et que la bibliotheque ne portait pas -- la linearisation et le
changement `t = tan(x/2)` -- sont exerces sur les memes exercices.
"""

from __future__ import annotations

import mpmath
import pytest
import sympy as sp

from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.substitution import Substitution
from polytech.maths.domain.integration.trigonometrie import (
    T_DEMI,
    lineariser,
    par_la_tangente_de_l_arc_moitie,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
A, B = sp.symbols("a b", positive=True)
sin, cos, sinh, cosh, exp, log, sqrt = (
    sp.sin,
    sp.cos,
    sp.sinh,
    sp.cosh,
    sp.exp,
    sp.log,
    sp.sqrt,
)

PRIMITIVES = {
    "1": (exp(sin(X) ** 2) * sin(2 * X), exp(sin(X) ** 2)),
    "2a": (cos(X) ** 5, (15 * sin(X) - 10 * sin(X) ** 3 + 3 * sin(X) ** 5) / 15),
    "2a'": (cos(X) ** 5, sin(5 * X) / 80 + 5 * sin(3 * X) / 48 + 5 * sin(X) / 8),
    "2b": (cosh(X) ** 3, sinh(X) + sinh(X) ** 3 / 3),
    "2b'": (cosh(X) ** 3, sinh(3 * X) / 12 + 3 * sinh(X) / 4),
    "2c": (cos(X) ** 4, (sin(4 * X) + 8 * sin(2 * X) + 12 * X) / 32),
    "2d": (sinh(X) ** 4, (sinh(4 * X) - 8 * sinh(2 * X) + 12 * X) / 32),
    "3": (X**3 * exp(X), (X**3 - 3 * X**2 + 6 * X - 6) * exp(X)),
    "4a": (log(X), X * log(X) - X),
    "4b": (X * log(X), X**2 * log(X) / 2 - X**2 / 4),
    "4c": (sp.asin(X), X * sp.asin(X) + sqrt(1 - X**2)),
    "5": (cosh(X) * sin(X), (sinh(X) * sin(X) - cosh(X) * cos(X)) / 2),
    "6": (1 / sin(X), log(sp.tan(X / 2))),
    "7": (sqrt(A**2 - X**2), A**2 / 2 * sp.asin(X / A) + X / 2 * sqrt(A**2 - X**2)),
    "8": (exp(2 * X) / sqrt(exp(X) + 1), R(2, 3) * sqrt(exp(X) + 1) * (exp(X) - 2)),
    "9a": (
        exp(A * X) * cos(B * X),
        exp(A * X) * (A * cos(B * X) + B * sin(B * X)) / (A**2 + B**2),
    ),
    "9b": (
        exp(A * X) * sin(B * X),
        exp(A * X) * (-B * cos(B * X) + A * sin(B * X)) / (A**2 + B**2),
    ),
    "10": (
        sqrt(X / (1 - X) ** 3),
        2 * sqrt(X / (1 - X)) - 2 * sp.atan(sqrt(X / (1 - X))),
    ),
    "11": (X**2 / sqrt(1 - X**2), (sp.asin(X) - X * sqrt(1 - X**2)) / 2),
    "12": (1 / (cos(X) + 2 * sin(X) + 3), sp.atan(sp.tan(X / 2) + 1)),
    "13": (sqrt(X) / sqrt(A**3 - X**3), R(2, 3) * sp.asin(sqrt(X**3 / A**3))),
    "14": (cosh(X) / (cosh(X) + sinh(X)), X / 2 + sinh(2 * X) / 4 - cosh(2 * X) / 4),
    "14'": (cosh(X) / (cosh(X) + sinh(X)), X / 2 - exp(-2 * X) / 4),
}
POINTS = (R(1, 5), R(2, 5), R(3, 5))
"""Trois points de `]0, 1[`, ou toutes les fonctions de la feuille sont definies."""


@pytest.mark.parametrize("numero", list(PRIMITIVES))
def test_chaque_primitive_officielle_redonne_son_integrande(numero: str) -> None:
    """Page 3 : deriver la reponse doit rendre la fonction a integrer."""
    integrande, primitive = PRIMITIVES[numero]
    ecart = (sp.diff(primitive, X) - integrande).subs({A: 2, B: 3})
    for point in POINTS:
        assert abs(sp.N(ecart.subs(X, point), 30)) < sp.Float("1e-20")


def test_les_deux_formes_d_une_meme_primitive_ne_different_que_d_une_constante() -> (
    None
):
    """Exercices 2 et 14 : « ou » relie deux ecritures de la meme primitive."""
    for premiere, seconde in (("2a", "2a'"), ("2b", "2b'"), ("14", "14'")):
        difference = PRIMITIVES[premiere][1] - PRIMITIVES[seconde][1]
        valeurs = {sp.N(difference.subs(X, point), 25) for point in POINTS}
        assert max(valeurs) - min(valeurs) < sp.Float("1e-18")


def test_la_linearisation_de_l_exercice_2() -> None:
    """`cos^5 t = (cos 5t + 5 cos 3t + 10 cos t) / 16` ; d'ou la seconde reponse."""
    linearise = lineariser(cos(X) ** 5)
    attendu = (cos(5 * X) + 5 * cos(3 * X) + 10 * cos(X)) / 16
    assert sp.simplify(linearise - attendu) == 0
    assert not linearise.has(sp.Pow)
    primitive = sp.integrate(linearise, X)
    assert sp.simplify(primitive - PRIMITIVES["2a'"][1]) == 0


def test_la_linearisation_d_une_puissance_paire() -> None:
    """`cos^4 t = (cos 4t + 4 cos 2t + 3) / 8`."""
    attendu = (cos(4 * X) + 4 * cos(2 * X) + 3) / 8
    assert sp.simplify(lineariser(cos(X) ** 4) - attendu) == 0


def test_le_changement_de_variable_de_l_exercice_1() -> None:
    """`u = sin^2 x`, `du = sin 2x dx` : il reste `int e^u du`."""
    substitution = Substitution(exp(sin(X) ** 2) * sin(2 * X), sin(X) ** 2)
    assert sp.simplify(substitution.differentielle() - sin(2 * X)) == 0
    assert substitution.integrale_transformee() == exp(substitution.nouvelle_variable())


def test_le_changement_de_variable_de_l_exercice_8() -> None:
    """`u = e^x` : `int u / sqrt(u + 1) du`."""
    substitution = Substitution(exp(2 * X) / sqrt(exp(X) + 1), exp(X))
    u = substitution.nouvelle_variable()
    assert sp.simplify(substitution.integrale_transformee() - u / sqrt(u + 1)) == 0


def test_les_integrations_par_parties_de_l_exercice_3() -> None:
    """Trois integrations par parties : `(x^3 - 3x^2 + 6x - 6) e^x`."""
    parties = IntegrationParParties(X**3, exp(X))
    assert sp.simplify(parties.resultat() - PRIMITIVES["3"][1]) == 0


def test_la_tangente_de_l_arc_moitie_de_l_exercice_6() -> None:
    """`t = tan(x/2)` : `dx / sin x` devient `dt / t`, d'ou `ln|tan(x/2)|`."""
    assert sp.simplify(par_la_tangente_de_l_arc_moitie(1 / sin(X)) - 1 / T_DEMI) == 0


def test_la_tangente_de_l_arc_moitie_de_l_exercice_12() -> None:
    """`dx / (cos x + 2 sin x + 3)` devient `dt / (t^2 + 2t + 2)`, puis `t = 1 + u`."""
    en_t = par_la_tangente_de_l_arc_moitie(1 / (cos(X) + 2 * sin(X) + 3))
    assert sp.simplify(en_t - 1 / (T_DEMI**2 + 2 * T_DEMI + 2)) == 0
    assert sp.simplify(sp.integrate(en_t, T_DEMI) - sp.atan(T_DEMI + 1)) == 0


def test_la_primitive_de_l_exercice_12_n_est_pas_continue_sur_r() -> None:
    """« Mais, au fait, ne cherchait-on pas une primitive sur R ? » (page 3).

    L'integrande est continu et positif partout ; la reponse saute pourtant
    de `pi` a chaque multiple impair de `pi`, ou `tan(x/2)` n'est pas definie.
    """
    primitive = PRIMITIVES["12"][1]
    a_gauche = sp.limit(primitive, X, sp.pi, "-")
    a_droite = sp.limit(primitive, X, sp.pi, "+")
    assert (a_gauche, a_droite) == (sp.pi / 2, -sp.pi / 2)


DEFINIES = {
    1: (cos(X) ** 4, 0, sp.pi / 2, 3 * sp.pi / 16),
    2: (sin(X) ** 2 * cos(X) ** 3, -sp.pi / 2, sp.pi / 2, R(4, 15)),
    3: (X**2 * cos(X), 0, sp.pi / 2, sp.pi**2 / 4 - 2),
    4: (X**2 * sin(X) * cos(X) ** 2, -sp.pi / 2, sp.pi / 2, 0),
    5: (sin(X) / (1 + cos(X) ** 2), 0, sp.pi / 2, sp.pi / 4),
    6: (1 / (1 + sin(X)), 0, sp.pi / 2, 1),
    7: (sin(X) ** 2 / (sin(X) + cos(X)), 0, sp.pi / 2, -log(sqrt(2) - 1) / sqrt(2)),
    8: (
        sin(2 * X) / sqrt(1 - R(1, 2) * sin(X)),
        0,
        sp.pi / 2,
        4 * (2 - (R(1, 2) + 2) * sqrt(1 - R(1, 2))) / (3 * R(1, 2) ** 2),
    ),
    9: (X * log(X), 0, 1, R(-1, 4)),
    10: (sp.asin(X), 0, 1, sp.pi / 2 - 1),
    11: (2 * X / ((1 + X**2) * (3 + X**2)), 0, 3, log(R(5, 2)) / 2),
    12: (
        X**2 * sp.atan(X) / (1 + X**2),
        0,
        1,
        sp.pi / 4 - sp.pi**2 / 32 - log(sqrt(2)),
    ),
    13: (sqrt(exp(X) - 1), 0, log(2), 2 - sp.pi / 2),
    14: (1 / (sqrt(X) - 1), 4, 9, 2 + 2 * log(2)),
    15: (
        X * exp(X) / sqrt(exp(X) + 1),
        0,
        1,
        4 * sqrt(2)
        - 2 * sqrt(sp.E + 1)
        + 4 * log((sqrt(sp.E + 1) + 1) / (sqrt(2) + 1))
        - 2,
    ),
    16: (
        log(1 - R(1, 4) * X**2) / X**2,
        0,
        1,
        R(1, 2) * log(abs((1 - R(1, 2)) / (1 + R(1, 2)))) - log(1 - R(1, 4)),
    ),
    17: (1 / (2 + sqrt(1 - X**2)), 0, 1, sp.pi / 6 * (3 - 4 / sqrt(3))),
    18: (1 / (X + sqrt(X**2 + 1)), -1, 1, log(1 + sqrt(2)) + sqrt(2)),
    19: (sqrt(1 + X**2), -1, 1, log(1 + sqrt(2)) + sqrt(2)),
}


@pytest.mark.parametrize("numero", list(DEFINIES))
def test_chaque_integrale_definie_a_la_valeur_annoncee(numero: int) -> None:
    """Page 4, « verifier les valeurs » : par quadrature, sans primitive.

    Les exercices 8 et 16 portent un parametre `a` ; il vaut ici 1/2.
    """
    integrande, bas, haut, annoncee = DEFINIES[numero]
    bornes = [mpmath.mpf(str(sp.N(borne, 30))) for borne in (bas, haut)]
    mesuree = mpmath.quad(sp.lambdify(X, integrande, "mpmath"), bornes)
    assert abs(mesuree - sp.N(annoncee, 20)) < mpmath.mpf("1e-9")


def test_la_tangente_de_l_arc_moitie_ne_vaut_que_pour_sin_et_cos() -> None:
    """`x sin x` n'est pas rationnel en `sin x` et `cos x` : `x` y subsiste."""
    with pytest.raises(ValueError, match="pas rationnel en sin x et cos x"):
        par_la_tangente_de_l_arc_moitie(X * sin(X))
