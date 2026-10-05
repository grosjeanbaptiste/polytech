"""Recueil de primitives, series a lettres : chaque reponse officielle controlee.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 61 a
63 (enonces) et 67 a 72 (solutions) du fichier, lues en image a pleine
resolution. Huit series : fractions rationnelles, exponentielles, trois
series diverses, fonctions rationnelles en sinus et cosinus, fonctions
irrationnelles, racines de trinomes, et la serie finale.

Meme controle que pour les exercices numerotes : la reponse est derivee et
comparee a l'enonce en deux points de son domaine.

Deux reponses officielles sont fausses, et le scan a ete relu avant de
l'ecrire : la primitive de `1/(x^2 + x + 1)^3` (un coefficient `3/8` pour
`9/32`) et celle de `1/(racine(x - p) racine(q - x))` (un signe).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.controle import ecart_de_derivation
from polytech.maths.domain.integration.hermite import inverse_d_une_puissance
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
sin, cos, tan, sqrt, ln, exp = sp.sin, sp.cos, sp.tan, sp.sqrt, sp.log, sp.exp
sh, ch, atan, absolu = sp.sinh, sp.cosh, sp.atan, sp.Abs
A, B, P, Q, W, H = 2, 3, 2, 5, 3, R(1, 2)
DEFAUT = (R(1, 3), R(6, 5))
U = 2 / sqrt(3) * (X + R(1, 2))
TRINOME = X**2 + X + 1
MOITIE = tan(X / 2)
CUBIQUE = ((X + 1) / (X - 1)) ** R(1, 3)
SIXIEME = X ** R(1, 6)
RAPPORT = sqrt((2 - X) / (X + 1))
EULER = sqrt(X**2 + 1) - X
L100 = ln(100)

Serie = dict[str, tuple[sp.Expr, sp.Expr, tuple[object, ...]]]

FRACTIONS: Serie = {
    "a": (
        1 / (X * (X**2 - 4)),
        -ln(absolu(X)) / 4 + ln(absolu(X - 2)) / 8 + ln(absolu(X + 2)) / 8,
        DEFAUT,
    ),
    "b": (1 / (X * (X**2 + 4)), ln(absolu(X)) / 4 - ln(X**2 + 4) / 8, DEFAUT),
    "c": (
        X / (X**3 - 1),
        ln(absolu(X - 1)) / 3 - ln(TRINOME) / 6 + sqrt(3) / 3 * atan(U),
        DEFAUT,
    ),
    "d": (
        (X**3 - 2 * X + 1) / (X**3 + 3 * X**2),
        X - 1 / (3 * X) - R(7, 9) * ln(absolu(X)) - R(20, 9) * ln(absolu(X + 3)),
        DEFAUT,
    ),
    "e": (
        1 / (X**4 - 1),
        ln(absolu(X - 1)) / 4 - ln(absolu(X + 1)) / 4 - atan(X) / 2,
        DEFAUT,
    ),
    "f": (
        X / (X**4 - 1),
        ln(absolu(X - 1)) / 4 + ln(absolu(X + 1)) / 4 - ln(X**2 + 1) / 4,
        DEFAUT,
    ),
    "g": (
        (X + 1) / (X**2 + 4 * X + 5) ** 2,
        -(X + 3) / (2 * (X**2 + 4 * X + 5)) - atan(X + 2) / 2,
        DEFAUT,
    ),
    "h": (
        1 / ((X + 2) * (X**2 + 1) ** 2),
        (2 * X + 1) / (10 * (X**2 + 1))
        + ln(absolu(X + 2)) / 25
        - ln(1 + X**2) / 50
        + R(7, 25) * atan(X),
        DEFAUT,
    ),
    "c1": (1 / TRINOME, 2 * sqrt(3) / 3 * atan(U), DEFAUT),
    "c2": (
        1 / TRINOME**2,
        8 * sqrt(3) / 9 * (atan(U) / 2 + R(3, 8) * U / TRINOME),
        DEFAUT,
    ),
}
EXPONENTIELLES: Serie = {
    "a": (1 / (1 + exp(X)), ln(exp(X) / (1 + exp(X))), DEFAUT),
    "b": (
        exp(-X) / (1 + exp(X)) ** 3,
        (-3 * exp(2 * X) - R(9, 2) * exp(X) - 1) / (exp(X) * (1 + exp(X)) ** 2)
        - 3 * X
        + 3 * ln(1 + exp(X)),
        DEFAUT,
    ),
    "c": (2**X / (4**X + 1), atan(2**X) / ln(2), DEFAUT),
    "d": (
        1 / (exp(6 * X) + exp(3 * X) - 2),
        (-3 * X / 2 + ln(absolu(exp(3 * X) - 1)) / 3 + ln(exp(3 * X) + 2) / 6) / 3,
        DEFAUT,
    ),
}
DIVERSES: Serie = {
    "1a+": (absolu(X), X**2 / 2, DEFAUT),
    "1a-": (absolu(X), -(X**2) / 2, (-R(1, 3), -R(6, 5))),
    "1b": (
        X**3 * 100 ** (-4 * X),
        -(100 ** (-4 * X))
        / (4 * L100)
        * (X**3 + 3 * X**2 / (4 * L100) + 3 * X / (8 * L100**2) + 3 / (32 * L100**3)),
        DEFAUT,
    ),
    "1c": (
        (X**3 + 4) * sin(3 * X),
        -cos(3 * X) / 3 * (X**3 + 4 - R(2, 3) * X) + sin(3 * X) / 3 * (X**2 - R(2, 9)),
        DEFAUT,
    ),
    "1d": (sin(X) * sh(X), (sin(X) * ch(X) - cos(X) * sh(X)) / 2, DEFAUT),
    "1e": (atan(X) / X**2, -atan(X) / X + ln(absolu(X)) - ln(1 + X**2) / 2, DEFAUT),
    "2a": (X / (2 - 7 * X**2), -ln(absolu(2 - 7 * X**2)) / 14, DEFAUT),
    "2b": (3 * X * sqrt(X**2 + 6), (X**2 + 6) ** R(3, 2), DEFAUT),
    "2c": (sin(X) ** 2 * cos(X) ** 3, sin(X) ** 3 / 3 - sin(X) ** 5 / 5, DEFAUT),
    "2d": (
        (X**2 + 1) / (X**3 - X),
        -ln(absolu(X)) + ln(absolu(X - 1)) + ln(absolu(X + 1)),
        DEFAUT,
    ),
    "2e": (
        1 / (X**2 - 4 * X + 2),
        ln(absolu((X - 2 - sqrt(2)) / (X - 2 + sqrt(2)))) / (2 * sqrt(2)),
        DEFAUT,
    ),
    "2f": (
        1 / (X * sqrt(2 + X - X**2)),
        ln(absolu((RAPPORT - sqrt(2)) / (RAPPORT + sqrt(2)))) / sqrt(2),
        DEFAUT,
    ),
    "2g": (1 / sqrt((X**2 + 3) ** 3), X / (3 * sqrt(3 + X**2)), DEFAUT),
    "2h": (
        X**2 / (X**4 - 1),
        ln(absolu(X - 1)) / 4 - ln(absolu(X + 1)) / 4 + atan(X) / 2,
        DEFAUT,
    ),
    "3a": (X**3 * atan(X), X**4 / 4 * atan(X) - (X**3 / 3 - X + atan(X)) / 4, DEFAUT),
    "3b": (X * exp(X) / (1 + X) ** 2, -X * exp(X) / (1 + X) + exp(X), DEFAUT),
    "3c": (tan(X) ** 3, 1 / (2 * cos(X) ** 2) + ln(absolu(cos(X))), DEFAUT),
    "3d": (
        sin(X) * cos(X) ** 2 / (1 + sin(X) ** 2),
        cos(X) + ln(sqrt(2) - cos(X)) / sqrt(2) - ln(sqrt(2) + cos(X)) / sqrt(2),
        DEFAUT,
    ),
    "3e": (
        1 / ((X - 1) * sqrt(4 - X**2)),
        (
            ln(absolu(1 - sqrt(3) * sqrt((2 - X) / (2 + X))))
            - ln(absolu(1 + sqrt(3) * sqrt((2 - X) / (2 + X))))
        )
        / sqrt(3),
        DEFAUT,
    ),
}
TRIGONOMETRIQUES: Serie = {
    "a": (
        1 / (2 + 5 * cos(X)),
        -ln(absolu((sqrt(7) - sqrt(3) * MOITIE) / (sqrt(7) + sqrt(3) * MOITIE)))
        / sqrt(21),
        DEFAUT,
    ),
    "b": (1 / cos(X), sp.atanh(sin(X)), DEFAUT),
    "c": (
        cos(X) / (A * cos(X) + B * sin(X)),
        R(B, A**2 + B**2) * ln(absolu(A + B * tan(X)))
        - R(B, 2 * (A**2 + B**2)) * ln(tan(X) ** 2 + 1)
        + R(A, A**2 + B**2) * X,
        DEFAUT,
    ),
    "d": (
        sin(X) ** 4 * cos(X) ** 2,
        (X / 2 - sin(4 * X) / 8 - sin(2 * X) ** 3 / 6) / 8,
        DEFAUT,
    ),
    "e": (
        cos(A * X) * cos(B * X),
        (sin((A - B) * X) / (A - B) + sin((A + B) * X) / (A + B)) / 2,
        DEFAUT,
    ),
    "f": (
        1 / (sin(X) + cos(X)),
        -(ln(absolu(MOITIE - 1 - sqrt(2))) - ln(absolu(MOITIE - 1 + sqrt(2))))
        / sqrt(2),
        DEFAUT,
    ),
    "g": (1 / (sin(X) ** 2 * cos(X)), -1 / sin(X) + sp.atanh(sin(X)), DEFAUT),
    "h": (
        (1 + tan(X)) / (1 - tan(X)),
        -ln(absolu(1 - tan(X))) + ln(1 + tan(X) ** 2) / 2,
        DEFAUT,
    ),
    "i": (
        cos(X) ** 6,
        (R(5, 2) * X + 2 * sin(2 * X) + R(3, 8) * sin(4 * X) - sin(2 * X) ** 3 / 6) / 8,
        DEFAUT,
    ),
    "j": (
        sin(W * X) * cos(W * X + H),
        -sin(H) * X / 2 - cos(2 * W * X + H) / (4 * W),
        DEFAUT,
    ),
    "k": (sin(X) / (1 - sin(X)), -2 / (MOITIE - 1) - X, DEFAUT),
    "l": (cos(X) ** 3 / (1 + sin(X) ** 2), -sin(X) + 2 * atan(sin(X)), DEFAUT),
    "m": (
        1 / (3 - tan(X) ** 2),
        ln((sqrt(3) + tan(X)) / (sqrt(3) - tan(X))) / (8 * sqrt(3)) + X / 4,
        (R(1, 3), R(4, 5)),
    ),
    "n": (1 / (1 + cos(X) + 5 * sin(X)), ln(absolu(1 + 5 * MOITIE)) / 5, DEFAUT),
    "o": (sin(X) ** 3, cos(X) ** 3 / 3 - cos(X), DEFAUT),
    "p": (
        sin(X) * sin(2 * X) * sin(3 * X),
        (-cos(2 * X) / 2 - cos(4 * X) / 4 + cos(6 * X) / 6) / 4,
        DEFAUT,
    ),
}
IRRATIONNELLES: Serie = {
    "a": (
        1 / (1 + sqrt(3 + 4 * X)),
        (sqrt(3 + 4 * X) - ln(1 + sqrt(3 + 4 * X))) / 2,
        DEFAUT,
    ),
    "b": (
        CUBIQUE,
        2 * CUBIQUE / ((X + 1) / (X - 1) - 1)
        - R(2, 3) * ln(absolu(CUBIQUE - 1))
        + ln(CUBIQUE**2 + CUBIQUE + 1) / 3
        + 2 / sqrt(3) * atan((2 * CUBIQUE + 1) / sqrt(3)),
        (2, R(7, 2)),
    ),
    "c": (
        1 / (sqrt(X) + X ** R(1, 3)),
        6 * (sqrt(X) / 3 - X ** R(1, 3) / 2 + SIXIEME - ln(SIXIEME + 1)),
        DEFAUT,
    ),
    "d": (
        (sqrt(X) - 1) / (X ** R(1, 3) + 1),
        6
        * (
            SIXIEME**7 / 7
            - SIXIEME**5 / 5
            - SIXIEME**4 / 4
            + SIXIEME**3 / 3
            + SIXIEME**2 / 2
            - SIXIEME
            - ln(X ** R(1, 3) + 1) / 2
            + atan(SIXIEME)
        ),
        DEFAUT,
    ),
}
TRINOMES: Serie = {
    "a+": (
        sqrt(X**2 + P**2),
        X / 2 * sqrt(X**2 + P**2)
        + R(P**2, 2) * ln(absolu(X / P + sqrt(X**2 + P**2) / P)),
        DEFAUT,
    ),
    "a-": (
        sqrt(X**2 - P**2),
        X / 2 * sqrt(X**2 - P**2)
        - R(P**2, 2) * ln(absolu(X / P + sqrt(X**2 - P**2) / P)),
        (3, 5),
    ),
    "b": (
        sqrt(P**2 - X**2),
        R(P**2, 2) * (sp.asin(X / P) + X / P * sqrt(1 - X**2 / P**2)),
        DEFAUT,
    ),
    "c": (1 / (X * sqrt(X**2 + 1)), -sp.acoth(sqrt(X**2 + 1)), DEFAUT),
    "d": (1 / (X**2 * sqrt(X**2 + 1)), -sqrt(X**2 + 1) / X, DEFAUT),
    "f": (
        1 / sqrt(2 * X**2 + X),
        ln(absolu(4 * X + 1 + sqrt((4 * X + 1) ** 2 - 1))) / sqrt(2),
        DEFAUT,
    ),
}
FINALE: Serie = {
    "a": (2 + tan(X) ** 2, X + tan(X), DEFAUT),
    "b": (
        atan(X) / (1 + X) ** 3,
        -atan(X) / (2 * (1 + X) ** 2)
        - 1 / (4 * (1 + X))
        - ln(1 + X**2) / 8
        + ln(absolu(1 + X)) / 4,
        DEFAUT,
    ),
    "c": (
        X**4 / sqrt(X**2 + 1),
        -(
            -1 / (4 * EULER**4)
            + EULER**4 / 4
            + 6 * ln(absolu(EULER))
            + 2 / EULER**2
            - 2 * EULER**2
        )
        / 16,
        DEFAUT,
    ),
    "d": (
        1 / (X**3 * sqrt(X - 1)),
        (3 * sqrt(X - 1) ** 3 + 5 * sqrt(X - 1)) / (4 * X**2)
        + R(3, 4) * atan(sqrt(X - 1)),
        (2, 5),
    ),
    "e": (
        1 / (sin(X) ** 4 + cos(X) ** 4),
        sqrt(2) / 2 * atan(tan(2 * X) / sqrt(2)),
        (R(1, 3), R(1, 2)),
    ),
    "f": (
        X * ln(1 + X**3),
        X**2 / 2 * ln(1 + X**3)
        - R(3, 4) * X**2
        - ln(absolu(X + 1)) / 2
        + ln(X**2 - X + 1) / 4
        + sqrt(3) / 2 * atan(2 / sqrt(3) * (X - R(1, 2))),
        DEFAUT,
    ),
    "g": (exp(X) ** R(1, 3), 3 * exp(X / 3), DEFAUT),
    "h": ((X + sin(X)) / (1 + cos(X)), X * MOITIE, DEFAUT),
    "i": (
        (X**2 + 1) ** 2 * sh(X),
        (X**2 + 1) ** 2 * ch(X)
        - 4 * (X**3 + X) * sh(X)
        + 4 * (3 * X**2 + 1) * ch(X)
        - 24 * X * sh(X)
        + 24 * ch(X),
        DEFAUT,
    ),
}
SERIES = {
    "fractions": FRACTIONS,
    "exponentielles": EXPONENTIELLES,
    "diverses": DIVERSES,
    "trigonometriques": TRIGONOMETRIQUES,
    "irrationnelles": IRRATIONNELLES,
    "trinomes": TRINOMES,
    "finale": FINALE,
}
CAS = [(serie, lettre) for serie, contenu in SERIES.items() for lettre in contenu]


@pytest.mark.parametrize(("serie", "lettre"), CAS)
def test_la_reponse_officielle_de_la_serie_est_une_primitive(
    serie: str, lettre: str
) -> None:
    """La derivee de la reponse redonne l'enonce."""
    enonce, reponse, points = SERIES[serie][lettre]
    assert ecart_de_derivation(reponse, enonce, points) < R(1, 10**20), (serie, lettre)


def test_la_primitive_officielle_du_cube_du_trinome_porte_un_coefficient_faux() -> None:
    """`int dx/(x^2+x+1)^3` : le terme en `1/(x^2+x+1)` a pour coefficient `9/32`.

    La solution imprime `3/8`, celui de `n = 2`. Second chemin : avec
    `t = 2(x + 1/2)/racine(3)`, `x^2 + x + 1 = 3(1 + t^2)/4`, et la recurrence
    `I_3 = 3/8 arctan t + 3/8 t/(1+t^2) + 1/4 t/(1+t^2)^2` donne `9/32` et `9/64`.
    """
    facteur = 32 * sqrt(3) / 27
    imprimee = facteur * (
        R(3, 8) * atan(U) + R(3, 8) * U / TRINOME + R(9, 64) * U / TRINOME**2
    )
    juste = facteur * (
        R(3, 8) * atan(U) + R(9, 32) * U / TRINOME + R(9, 64) * U / TRINOME**2
    )
    enonce = 1 / TRINOME**3
    assert ecart_de_derivation(juste, enonce, DEFAUT) < R(1, 10**20)
    assert ecart_de_derivation(imprimee, enonce, DEFAUT) > R(1, 100)
    recurrence = facteur * inverse_d_une_puissance(3).subs(X, U)
    assert ecart_de_derivation(recurrence, enonce, DEFAUT) < R(1, 10**20)


def test_la_primitive_officielle_de_la_serie_des_trinomes_e_a_le_mauvais_signe() -> (
    None
):
    """`1/(racine(x - p) racine(q - x))`, qui demande `p < x < q`.

    La solution imprime `arcsin(2/(p - q) (x - (p + q)/2))`, dont la derivee
    est l'oppose de l'enonce ; c'est `2/(q - p)`.
    """
    enonce = 1 / (sqrt(X - P) * sqrt(Q - X))
    milieu = X - R(P + Q, 2)
    imprimee = sp.asin(R(2, P - Q) * milieu)
    juste = sp.asin(R(2, Q - P) * milieu)
    points = (3, 4)
    assert ecart_de_derivation(juste, enonce, points) < R(1, 10**20)
    assert ecart_de_derivation(-imprimee, enonce, points) < R(1, 10**20)
    assert ecart_de_derivation(imprimee, enonce, points) > R(1, 10)
