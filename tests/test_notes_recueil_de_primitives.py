"""Recueil de primitives (exercices 12) : chaque reponse officielle controlee.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 60 a
72 du fichier (un scan, lu en image a pleine resolution) : les enonces du
« Chapter 1 Primitives » et les « Solutions du chapitre XII ».

Chaque reponse est derivee et comparee a son enonce, en deux points du
domaine et avec 30 chiffres : une primitive fausse ne peut pas passer, et
une reponse juste ecrite autrement que ne le ferait un calcul formel passe.
Les parametres `a`, `b`, `p`, `n` recoivent des valeurs.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.controle import ecart_de_derivation
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
sin, cos, tan, sqrt, ln, exp = sp.sin, sp.cos, sp.tan, sp.sqrt, sp.log, sp.exp
sh, ch, atan = sp.sinh, sp.cosh, sp.atan
A, B, P, N = 2, 3, 5, 3
U = X + 1
QUATRIEME = (X + 1) ** R(1, 4)
DENOMINATEUR_40 = 16**2 + 30**2

NUMEROTEES: dict[int, tuple[sp.Expr, sp.Expr]] = {
    1: (
        (1 + 2 * X + X**4) ** 2,
        X + 2 * X**2 + R(4, 3) * X**3 + R(2, 5) * X**5 + R(2, 3) * X**6 + X**9 / 9,
    ),
    2: (exp(-X) * (1 + 5 * X**2), -exp(-X) * (5 * X**2 + 10 * X + 11)),
    3: (sqrt(A + B * X), R(2, 3 * B) * (A + B * X) ** R(3, 2)),
    4: (
        X * sqrt(A + B * X),
        R(2, B**2) * ((A + B * X) ** R(5, 2) / 5 - R(A, 3) * (A + B * X) ** R(3, 2)),
    ),
    5: (1 / (3 * X**2 + 7), atan(sqrt(R(3, 7)) * X) / sqrt(21)),
    6: (tan(X), -ln(cos(X))),
    7: (sqrt(1 - cos(X)), -2 * sqrt(2) * cos(X / 2)),
    8: (sqrt(1 + cos(X)), 2 * sqrt(2) * sin(X / 2)),
    9: (3 ** (5 * X), 3 ** (5 * X) / ln(3**5)),
    10: (exp(X) / (1 + exp(2 * X)), atan(exp(X))),
    11: (sin(X) * cos(X) ** N, -(cos(X) ** (N + 1)) / (N + 1)),
    12: (tan(X) ** 2, tan(X) - X),
    13: (
        (4 * X + 5) ** 18 * X,
        ((5 + 4 * X) ** 20 / 20 - R(5, 19) * (5 + 4 * X) ** 19) / 16,
    ),
    14: (tan(ln(X)) / X, -ln(cos(ln(X)))),
    15: (1 / sin(X) ** 2, -1 / tan(X)),
    16: (
        exp(A * X) * cos(P * X),
        R(P**2, P**2 + A**2) * exp(A * X) * (sin(P * X) / P + R(A, P**2) * cos(P * X)),
    ),
    17: (X / (5 * X**2 + 18), ln(5 * X**2 + 18) / 10),
    18: (exp(X) * (3 + 10 * exp(X)) ** R(1, 5), (3 + 10 * exp(X)) ** R(6, 5) / 12),
    19: (
        exp(A * X) * sin(P * X),
        R(P**2, P**2 + A**2) * exp(A * X) * (-cos(P * X) / P + R(A, P**2) * sin(P * X)),
    ),
    20: (1 / (1 + B * exp(P * X)), -ln(exp(-P * X) + B) / P),
    21: (sin(X) ** 2, (X - sin(2 * X) / 2) / 2),
    22: (
        X**4 * exp(X / 2),
        2 * exp(X / 2) * (X**4 - 8 * X**3 + 48 * X**2 - 192 * X + 384),
    ),
    23: (X * sh(X), X * ch(X) - sh(X)),
    24: (1 / ch(X) ** 2, sp.tanh(X)),
    25: (
        exp(X / 2) * sin(5 * X),
        R(100, 101) * exp(X / 2) * (-cos(5 * X) / 5 + sin(5 * X) / 50),
    ),
    26: (sin(ln(X)), X * (sin(ln(X)) - cos(ln(X))) / 2),
    27: (sh(X) ** 2, (sh(2 * X) / 2 - X) / 2),
    28: ((X + 1) / (X**2 + 4 * X + 5), ln(X**2 + 4 * X + 5) / 2 - atan(X + 2)),
    29: (exp(sqrt(X)), 2 * exp(sqrt(X)) * (sqrt(X) - 1)),
    30: (atan(X), X * atan(X) - ln(1 + X**2) / 2),
    31: (X**P * ln(A * X), X ** (P + 1) / (P + 1) * (ln(A * X) - R(1, P + 1))),
    32: (
        X**2 * cos(X) ** 2,
        X**3 / 6 + X**2 / 4 * sin(2 * X) + X / 4 * cos(2 * X) - sin(2 * X) / 8,
    ),
    33: (sp.asin(X), X * sp.asin(X) + sqrt(1 - X**2)),
    34: (exp(X) / sqrt(1 + exp(X)), 2 * sqrt(1 + exp(X))),
    35: (
        X**2 * (1 + X**3) ** R(54, 100),
        (1 + X**3) ** R(154, 100) / (3 * R(154, 100)),
    ),
    36: (1 / (X**2 + X + 1), 2 * sqrt(3) / 3 * atan(2 / sqrt(3) * (X + R(1, 2)))),
    37: (U**3 * sin(X), sin(X) * (3 * U**2 - 6) - cos(X) * (U**3 - 6 * U)),
    38: (1 / (sqrt(U) - QUATRIEME), 4 * (sqrt(U) / 2 + QUATRIEME + ln(QUATRIEME - 1))),
    39: ((X**2 + 1) * atan(X), (X**3 / 3 + X) * atan(X) - X**2 / 6 - ln(1 + X**2) / 3),
    40: (
        X**2 * sin(3 * X) * exp(-5 * X),
        exp(-5 * X)
        / 34
        * (
            (-5 * cos(3 * X) + 3 * sin(3 * X)) * (R(3, 17) * X + R(60, DENOMINATEUR_40))
            + (-3 * cos(3 * X) - 5 * sin(3 * X))
            * (X**2 + R(5, 17) * X + R(32, DENOMINATEUR_40))
        ),
    ),
    41: (sh(X) * sh(3 * X), (sh(4 * X) - 2 * sh(2 * X)) / 8),
    42: (U**3 * ch(X), U**3 * sh(X) - 3 * U**2 * ch(X) + 6 * U * sh(X) - 6 * ch(X)),
    43: (1 / (X**2 + 4), atan(X / 2) / 2),
    44: (1 / sqrt(1 - 3 * X**2), sp.asin(sqrt(3) * X) / sqrt(3)),
    45: (ln(X) ** N, X * (ln(X) ** 3 - 3 * ln(X) ** 2 + 6 * ln(X) - 6)),
    46: (
        X**4 * ln(X) ** 3,
        X**5 / 5 * (ln(X) ** 3 - R(3, 5) * ln(X) ** 2 + R(6, 25) * ln(X) - R(6, 125)),
    ),
    47: (1 / cos(5 * X) ** 2, tan(5 * X) / 5),
    48: (1 / sqrt(1 + X**2), sp.asinh(X)),
    49: (X**N * exp(-X), -exp(-X) * (X**3 + 3 * X**2 + 6 * X + 6)),
    50: (cos(X) * sh(3 * X), R(3, 10) * cos(X) * ch(3 * X) + sin(X) * sh(3 * X) / 10),
    51: (X**3 * exp(-(X**2)), -exp(-(X**2)) * (X**2 + 1) / 2),
    52: (cos(ln(X)), X / 2 * (cos(ln(X)) + sin(ln(X)))),
    53: (X**2 * exp(sin(X**3)) * cos(X**3), exp(sin(X**3)) / 3),
    54: (ln(X + sqrt(1 + X**2)), X * sp.asinh(X) - sqrt(1 + X**2)),
    55: (
        2**X * sin(3 * X),
        9
        / (9 + ln(2) ** 2)
        * (-(2**X) * cos(3 * X) / 3 + ln(2) / 9 * 2**X * sin(3 * X)),
    ),
    56: (ln(X) ** 2, X * ln(X) ** 2 - 2 * X * ln(X) + 2 * X),
}
"""Numero de l'exercice : (enonce, reponse officielle)."""

POINTS = {38: (R(3, 2), 4), 44: (R(1, 5), R(2, 5)), 33: (R(1, 5), R(2, 5))}
"""Les exercices dont le domaine n'accueille pas les points par defaut."""


@pytest.mark.parametrize("numero", sorted(NUMEROTEES))
def test_la_reponse_officielle_est_une_primitive_de_l_enonce(numero: int) -> None:
    """La derivee de la reponse redonne l'enonce."""
    enonce, reponse = NUMEROTEES[numero]
    points = POINTS.get(numero, (R(1, 3), R(6, 5)))
    assert ecart_de_derivation(reponse, enonce, points) < R(1, 10**20), numero


def test_le_controle_refuse_une_primitive_fausse() -> None:
    """Un signe retourne dans la reponse 2 : l'ecart n'est plus nul."""
    enonce, reponse = NUMEROTEES[2]
    assert ecart_de_derivation(-reponse, enonce, (R(1, 3), R(6, 5))) > R(1, 10)


def test_le_controle_demande_au_moins_un_point() -> None:
    with pytest.raises(ValueError, match="aucun point de controle"):
        ecart_de_derivation(X**2, 2 * X, ())


def test_un_seul_point_ou_l_ecart_s_annule_ne_suffit_pas() -> None:
    """`F' - f = x - 1/3` s'annule en `1/3` : c'est le plus grand ecart qui compte."""
    fausse = X**2 / 2 - X / 3
    ecart = ecart_de_derivation(fausse, sp.Integer(0), (R(1, 3), R(6, 5)))
    assert abs(ecart - R(13, 15)) < R(1, 10**20)
