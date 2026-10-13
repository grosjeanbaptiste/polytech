"""Les enonces du recueil de primitives, exercices 8 a 11.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 62
et 63 du PDF, lues sur le scan. Donnees seules : une entree par primitive.
Le 10 a) porte un double signe, d'ou ses deux entrees ; le 8 j) est en `t`
dans l'enonce et en `x` ici.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application.recueil_primitives_symboles import (
    A,
    B,
    H,
    P,
    Q,
    R,
    W,
    X,
)

sin, cos, tan, exp, sqrt, ln = sp.sin, sp.cos, sp.tan, sp.exp, sp.sqrt, sp.log
atan, sinh, cosh = sp.atan, sp.sinh, sp.cosh

EXERCICES_8_A_11: Final[dict[str, sp.Expr]] = {
    "8a": 1 / (2 + 5 * cos(X)),
    "8b": 1 / cos(X),
    "8c": cos(X) / (A * cos(X) + B * sin(X)),
    "8d": sin(X) ** 4 * cos(X) ** 2,
    "8e": cos(A * X) * cos(B * X),
    "8f": 1 / (sin(X) + cos(X)),
    "8g": 1 / (sin(X) ** 2 * cos(X)),
    "8h": (1 + tan(X)) / (1 - tan(X)),
    "8i": cos(X) ** 6,
    "8j": sin(W * X) * cos(W * X + H),
    "8k": sin(X) / (1 - sin(X)),
    "8l": cos(X) ** 3 / (1 + sin(X) ** 2),
    "8m": 1 / (3 - tan(X) ** 2),
    "8n": 1 / (1 + cos(X) + 5 * sin(X)),
    "8o": sin(X) ** 3,
    "8p": sin(X) * sin(2 * X) * sin(3 * X),
    "9a": 1 / (1 + sqrt(3 + 4 * X)),
    "9b": sp.cbrt((X + 1) / (X - 1)),
    "9c": 1 / (sqrt(X) + sp.cbrt(X)),
    "9d": (sqrt(X) - 1) / (sp.cbrt(X) + 1),
    "10aPlus": sqrt(X**2 + P**2),
    "10aMoins": sqrt(X**2 - P**2),
    "10b": sqrt(P**2 - X**2),
    "10c": 1 / (X * sqrt(X**2 + 1)),
    "10d": 1 / (X**2 * sqrt(X**2 + 1)),
    "10e": 1 / (sqrt(X - P) * sqrt(Q - X)),
    "10f": 1 / sqrt(2 * X**2 + X),
    "11a": 2 + tan(X) ** 2,
    "11b": atan(X) / (1 + X) ** 3,
    "11c": X**4 / sqrt(X**2 + 1),
    "11d": 1 / (X**3 * sqrt(X - 1)),
    "11e": 1 / (sin(X) ** 4 + cos(X) ** 4),
    "11f": X * ln(1 + X**3),
    "11g": sp.cbrt(exp(X)),
    "11h": (X + sin(X)) / (1 + cos(X)),
    "11i": (X**2 + 1) ** 2 * sinh(X),
}

POINTS_PARTICULIERS: Final[dict[str, tuple[sp.Expr, ...]]] = {
    "9b": (R(6, 5), R(3, 2), R(19, 10)),
    "11d": (R(6, 5), R(3, 2), R(19, 10)),
    "10b": (R(1, 50), R(1, 20), R(2, 25)),
}
"""Les primitives dont le domaine ne contient pas les points de controle communs :
`x > 1` pour deux d'entre elles, `|x| < p` pour la troisieme."""
