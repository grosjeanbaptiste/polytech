"""Recueil de primitives, exercices 2, 4 et 5.

Fractions rationnelles, exponentielles -- ou poser `u = e^x` rend
l'integrande rationnel --, puis cinq fonctions diverses.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.primitives_a_exs_01 import parties
from polytech.maths.application.recueil_primitives_enonces import (
    EXERCICES_SUIVANTS as F,
)
from polytech.maths.application.recueil_primitives_enonces import K, X
from polytech.maths.application.recueil_primitives_gestes import (
    U,
    changement,
    immediate,
    terme_a_terme,
)
from polytech.maths.application.recueil_primitives_parties import par_parties
from polytech.maths.application.recueil_primitives_sommes import fractions_simples
from polytech.maths.application.resolution_exs_01 import Resolution, par_equation

sin, cos, exp, ln = sp.sin, sp.cos, sp.exp, sp.log
TRINOME = X**2 + 4 * X + 5


def rationnelles() -> dict[str, Resolution]:
    """Exercice 2 : huit fractions rationnelles."""
    en_deux = (2 * X + 4) / (2 * TRINOME**2) - 1 / ((X + 2) ** 2 + 1) ** 2
    resolutions = {
        cle: fractions_simples(F[cle]) for cle in ["2a", "2b", "2c", "2e", "2f", "2h"]
    }
    division = sp.apart(F["2d"], X)
    resolutions["2d"] = terme_a_terme(F["2d"], division, "division, puis fractions")
    resolutions["2g"] = terme_a_terme(F["2g"], en_deux, "decomposition")
    return resolutions


def exponentielles() -> dict[str, Resolution]:
    """Exercice 4 : `u = e^x` -- ou `e^-x`, `2^x`, `e^3x` -- rend tout rationnel."""
    return {
        "4a": changement(F["4a"], -1 / U, exp(-X) + 1),
        "4b": changement(F["4b"], 1 / (U**2 * (1 + U) ** 3), exp(X)),
        "4c": changement(F["4c"], 1 / (ln(2) * (1 + U**2)), 2**X),
        "4d": changement(F["4d"], 1 / (3 * U * (U - 1) * (U + 2)), exp(3 * X)),
    }


def diverses() -> dict[str, Resolution]:
    """Exercice 5 : une valeur absolue, trois produits, un arc tangente."""
    return {
        "5a": immediate(F["5a"], X * sp.Abs(X) / 2),
        "5b": parties(X**3, exp(-K * X), gestes=3),
        "5c": parties(X**3 + 4, sin(3 * X), gestes=3),
        "5d": par_equation((sin(X), sp.sinh(X)), (cos(X), sp.cosh(X))),
        "5e": par_parties(sp.atan(X), 1 / X**2),
    }
