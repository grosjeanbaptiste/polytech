"""Recueil de primitives, exercices 9, 10 et 11.

Les radicaux. Une racine d'une fonction affine ou d'une homographie se
prend pour variable ; la racine d'un trinome se traite par une fonction
trigonometrique ou hyperbolique, qui en fait un carre parfait.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.primitives_a_exs_01 import parties
from polytech.maths.application.recueil_primitives_enonces import (
    EXERCICES_SUIVANTS as F,
)
from polytech.maths.application.recueil_primitives_enonces import (
    POINTS_PARTICULIERS as POINTS,
)
from polytech.maths.application.recueil_primitives_enonces import P, Q, R, X
from polytech.maths.application.recueil_primitives_gestes import (
    T,
    U,
    changement,
    changement_inverse,
)
from polytech.maths.application.recueil_primitives_parties import par_parties
from polytech.maths.application.recueil_primitives_sommes import (
    par_morceaux,
    reconnue,
)

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution

sin, cos, tan, exp, sqrt, ln = sp.sin, sp.cos, sp.tan, sp.exp, sp.sqrt, sp.log
sinh, cosh, atan = sp.sinh, sp.cosh, sp.atan
HOMOGRAPHIE = (X + 1) / (X - 1)


def exercice_9() -> dict[str, Resolution]:
    """Quatre irrationnelles simples : le radical devient la variable."""
    inverse = (T**3 + 1) / (T**3 - 1)
    en_t = -6 * T**3 / (T**3 - 1) ** 2
    sixieme = X ** R(1, 6)
    return {
        "9a": changement_inverse(
            F["9a"], (T**2 - 3) / 4, sqrt(3 + 4 * X), T / (2 * (1 + T))
        ),
        "9b": changement_inverse(
            F["9b"], inverse, sp.cbrt(HOMOGRAPHIE), en_t, POINTS["9b"]
        ),
        "9c": changement_inverse(F["9c"], T**6, sixieme, 6 * T**3 / (T + 1)),
        "9d": changement_inverse(
            F["9d"], T**6, sixieme, 6 * T**5 * (T**3 - 1) / (T**2 + 1)
        ),
    }


def exercice_10() -> dict[str, Resolution]:
    """La racine carree d'un trinome du second degre."""
    centre = (2 * X - P - Q) / (Q - P)
    return {
        "10aPlus": changement_inverse(
            F["10aPlus"], P * sinh(T), sp.asinh(X / P), P**2 * cosh(T) ** 2
        ),
        "10aMoins": changement_inverse(
            F["10aMoins"], P * cosh(T), sp.acosh(X / P), P**2 * sinh(T) ** 2
        ),
        "10b": changement_inverse(
            F["10b"], P * sin(T), sp.asin(X / P), P**2 * cos(T) ** 2, POINTS["10b"]
        ),
        "10c": changement(F["10c"], 1 / (U**2 - 1), sqrt(X**2 + 1)),
        "10d": changement_inverse(F["10d"], 1 / T, 1 / X, -T / sqrt(1 + T**2)),
        "10e": changement(F["10e"], 1 / sqrt(1 - U**2), centre),
        "10f": changement(F["10f"], 1 / (sqrt(2) * sqrt(U**2 - 1)), 4 * X + 1),
    }


def exercice_11() -> dict[str, Resolution]:
    """Neuf primitives de revision."""
    separe = ((sp.Integer(1), X), (1 + tan(X) ** 2, tan(X)))
    produit = X / (2 * cos(X / 2) ** 2) + tan(X / 2)
    return {
        "11a": par_morceaux(F["11a"], separe, "decomposition"),
        "11b": par_parties(atan(X), 1 / (1 + X) ** 3),
        "11c": changement_inverse(F["11c"], sinh(T), sp.asinh(X), sinh(T) ** 4),
        "11d": changement_inverse(
            F["11d"], T**2 + 1, sqrt(X - 1), 2 / (T**2 + 1) ** 3, POINTS["11d"]
        ),
        "11e": changement(F["11e"], 1 / (2 + U**2), tan(2 * X)),
        "11f": par_parties(ln(1 + X**3), X),
        "11g": changement(F["11g"], 3 * exp(U), X / 3),
        "11h": reconnue(F["11h"], produit, X * tan(X / 2), "derivee d'un produit"),
        "11i": parties((X**2 + 1) ** 2, sinh(X), gestes=4),
    }
