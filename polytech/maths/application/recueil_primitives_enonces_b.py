"""Les enonces du recueil de primitives, exercices 2 a 7.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 61
et 62 du PDF, lues sur le scan. Donnees seules : une entree par primitive.
Le 3 est une recurrence, traitee a part.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application.recueil_primitives_symboles import X

sin, cos, tan, exp, sqrt, ln = sp.sin, sp.cos, sp.tan, sp.exp, sp.sqrt, sp.log
atan, sinh, cosh = sp.atan, sp.sinh, sp.cosh

EXERCICES_2_A_7: Final[dict[str, sp.Expr]] = {
    "2a": 1 / (X * (X**2 - 4)),
    "2b": 1 / (X * (X**2 + 4)),
    "2c": X / (X**3 - 1),
    "2d": (X**3 - 2 * X + 1) / (X**3 + 3 * X**2),
    "2e": 1 / (X**4 - 1),
    "2f": X / (X**4 - 1),
    "2g": (X + 1) / (X**2 + 4 * X + 5) ** 2,
    "2h": 1 / ((X + 2) * (X**2 + 1) ** 2),
    "4a": 1 / (1 + exp(X)),
    "4b": exp(-X) / (1 + exp(X)) ** 3,
    "4c": 2**X / (4**X + 1),
    "4d": 1 / (exp(6 * X) + exp(3 * X) - 2),
    "5a": sp.Abs(X),
    "5b": X**3 * 100 ** (-4 * X),
    "5c": (X**3 + 4) * sin(3 * X),
    "5d": sin(X) * sinh(X),
    "5e": atan(X) / X**2,
    "6a": X / (2 - 7 * X**2),
    "6b": 3 * X * sqrt(X**2 + 6),
    "6c": sin(X) ** 2 * cos(X) ** 3,
    "6d": (X**2 + 1) / (X**3 - X),
    "6e": 1 / (X**2 - 4 * X + 2),
    "6f": 1 / (X * sqrt(2 + X - X**2)),
    "6g": 1 / sqrt((X**2 + 3) ** 3),
    "6h": X**2 / (X**4 - 1),
    "7a": X**3 * atan(X),
    "7b": X * exp(X) / (1 + X) ** 2,
    "7c": tan(X) ** 3,
    "7d": sin(X) * cos(X) ** 2 / (1 + sin(X) ** 2),
    "7e": 1 / ((X - 1) * sqrt(4 - X**2)),
}
