"""Recueil de primitives, exercices 6 et 7.

Deux radicaux y demandent un changement `x = phi(t)` : au 6 f) et au
7 e), le trinome sous la racine se factorise, et `t` est la racine du
quotient de ses deux facteurs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.recueil_primitives_enonces import (
    EXERCICES_SUIVANTS as F,
)
from polytech.maths.application.recueil_primitives_enonces import X
from polytech.maths.application.recueil_primitives_gestes import (
    T,
    U,
    changement,
    changement_inverse,
    terme_a_terme,
)
from polytech.maths.application.recueil_primitives_parties import par_parties
from polytech.maths.application.recueil_primitives_sommes import fractions_simples

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution

sin, cos, exp, sqrt, atan = sp.sin, sp.cos, sp.exp, sp.sqrt, sp.atan
RACINES = (X - 2 - sqrt(2), X - 2 + sqrt(2))
"""Les deux facteurs de `x^2 - 4x + 2`, dont les racines sont `2 +- racine(2)`."""


def exercice_6() -> dict[str, Resolution]:
    """Huit primitives : trois changements simples, trois fractions, deux radicaux."""
    simples = (1 / RACINES[0] - 1 / RACINES[1]) / (2 * sqrt(2))
    en_t = (2 - T**2) / (T**2 + 1)
    return {
        "6a": changement(F["6a"], -1 / (14 * U), 2 - 7 * X**2),
        "6b": changement(F["6b"], 3 * sqrt(U) / 2, X**2 + 6),
        "6c": changement(F["6c"], U**2 * (1 - U**2), sin(X)),
        "6d": fractions_simples(F["6d"]),
        "6e": terme_a_terme(F["6e"], simples, "decomposition en fractions simples"),
        "6f": changement_inverse(
            F["6f"], en_t, sqrt((2 - X) / (X + 1)), 2 / (T**2 - 2)
        ),
        "6g": changement_inverse(
            F["6g"], sqrt(3) * sp.tan(T), atan(X / sqrt(3)), cos(T) / 3
        ),
        "6h": fractions_simples(F["6h"]),
    }


def exercice_7() -> dict[str, Resolution]:
    """Cinq primitives : deux par parties, deux par `u = cos x`, un radical."""
    en_t = 2 * (1 - T**2) / (1 + T**2)
    return {
        "7a": par_parties(atan(X), X**3),
        "7b": par_parties(X * exp(X), 1 / (1 + X) ** 2),
        "7c": changement(F["7c"], -(1 - U**2) / U**3, cos(X)),
        "7d": changement(F["7d"], -(U**2) / (2 - U**2), cos(X)),
        "7e": changement_inverse(
            F["7e"], en_t, sqrt((2 - X) / (2 + X)), -2 / (1 - 3 * T**2)
        ),
    }
