"""Les enonces du recueil de primitives du syllabus de 2011.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, chapitre
« Primitives », pages 60 a 63 du PDF. Le document est un scan : chaque
integrande a ete lu sur la page. Ce fichier ne porte que des donnees, une
entree par primitive demandee.

Les parametres `a`, `b`, `p` sont pris strictement positifs et `n` entier
positif : c'est le cas general de chaque enonce, et les cas particuliers
que le syllabus signale (`b = 0`, `p = -1`) sont dits dans le corrige.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application.resolution_exs_01 import A, B
from polytech.maths.domain.symboles import ENTIER as N
from polytech.maths.domain.symboles import VARIABLE as X

__all__ = ["EXERCICE_UN", "A", "B", "N", "P", "R", "X"]

P: Final = sp.Symbol("p", positive=True)
R = sp.Rational
ln, sin, cos, tan, exp, sqrt = sp.log, sp.sin, sp.cos, sp.tan, sp.exp, sp.sqrt
atan, asin, sinh, cosh = sp.atan, sp.asin, sp.sinh, sp.cosh

EXERCICE_UN: Final[dict[int, sp.Expr]] = {
    1: (1 + 2 * X + X**4) ** 2,
    2: exp(-X) * (1 + 5 * X**2),
    3: sqrt(A + B * X),
    4: X * sqrt(A + B * X),
    5: 1 / (3 * X**2 + 7),
    6: tan(X),
    7: sqrt(1 - cos(X)),
    8: sqrt(1 + cos(X)),
    9: 3 ** (5 * X),
    10: exp(X) / (1 + exp(2 * X)),
    11: sin(X) * cos(X) ** N,
    12: tan(X) ** 2,
    13: (4 * X + 5) ** 18 * X,
    14: tan(ln(X)) / X,
    15: 1 / sin(X) ** 2,
    16: exp(A * X) * cos(P * X),
    17: X / (5 * X**2 + 18),
    18: exp(X) * (3 + 10 * exp(X)) ** R(1, 5),
    19: exp(A * X) * sin(P * X),
    20: 1 / (1 + B * exp(P * X)),
    21: sin(X) ** 2,
    22: X**4 * exp(X / 2),
    23: X * sinh(X),
    24: 1 / cosh(X) ** 2,
    25: exp(X / 2) * sin(5 * X),
    26: sin(ln(X)),
    27: sinh(X) ** 2,
    28: (X + 1) / (X**2 + 4 * X + 5),
    29: exp(sqrt(X)),
    30: atan(X),
    31: X**P * ln(A * X),
    32: X**2 * cos(X) ** 2,
    33: asin(X),
    34: exp(X) / sqrt(1 + exp(X)),
    35: X**2 * (1 + X**3) ** R(27, 50),
    36: 1 / (X**2 + X + 1),
    37: (X + 1) ** 3 * sin(X),
    38: 1 / (sqrt(X + 1) - (X + 1) ** R(1, 4)),
    39: (X**2 + 1) * atan(X),
    40: X**2 * sin(3 * X) * exp(-5 * X),
    41: sinh(X) * sinh(3 * X),
    42: (X + 1) ** 3 * cosh(X),
    43: 1 / (X**2 + 4),
    44: 1 / sqrt(1 - 3 * X**2),
    45: ln(X) ** N,
    46: X**4 * ln(X) ** 3,
    47: 1 / cos(5 * X) ** 2,
    48: 1 / sqrt(1 + X**2),
    49: X**N * exp(-X),
    50: cos(X) * sinh(3 * X),
    51: X**3 * exp(-(X**2)),
    52: cos(ln(X)),
    53: X**2 * exp(sin(X**3)) * cos(X**3),
    54: ln(X + sqrt(1 + X**2)),
    55: 2**X * sin(3 * X),
    56: ln(X) ** 2,
}
"""Exercice 1 : « determiner les primitives par les methodes classiques »."""
