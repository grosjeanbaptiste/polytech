"""Recueil de primitives, exercice 1 : les primitives 21 a 40.

Le 40 se traite par l'exponentielle complexe : `sin 3x` est la partie
imaginaire de `e^(3ix)`, et deux integrations par parties de
`x^2 e^(cx)`, avec `c = -5 + 3i`, remplacent les six qu'il faudrait
dans les reels.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application.primitives_a_exs_01 import parties
from polytech.maths.application.recueil_primitives_enonces import (
    EXERCICE_UN as F,
)
from polytech.maths.application.recueil_primitives_enonces import A, P, R, X
from polytech.maths.application.recueil_primitives_gestes import (
    T,
    U,
    changement,
    changement_inverse,
    immediate,
    par_parties,
    terme_a_terme,
)
from polytech.maths.application.resolution_exs_01 import Resolution, par_equation

sin, cos, exp, sqrt, ln = sp.sin, sp.cos, sp.exp, sp.sqrt, sp.log
atan, asin, sinh, cosh = sp.atan, sp.asin, sp.sinh, sp.cosh
LINEARISATION = "linearisation"
COMPLEXE: Final = sp.Symbol("c")
"""`c = -5 + 3i`, l'exposant complexe de l'exercice 40."""
EXPOSANT_40: Final = -5 + 3 * sp.I


def par_l_exponentielle_complexe() -> Resolution:
    """`x^2 sin(3x) e^(-5x) = Im(x^2 e^(cx))`, primitive par parties en `c`."""
    en_c = exp(COMPLEXE * X) * (X**2 / COMPLEXE - 2 * X / COMPLEXE**2 + 2 / COMPLEXE**3)
    complexe = sp.expand_complex(en_c.subs(COMPLEXE, EXPOSANT_40))
    primitive = sp.factor_terms(sp.simplify(sp.im(complexe)))
    return Resolution(F[40], "exponentielle complexe", (en_c,), primitive)


def resolutions() -> dict[int, Resolution]:
    """Les primitives 21 a 40, chacune par son geste."""
    carre = (2 * X + 4) / (2 * (X**2 + 4 * X + 5)) - 1 / ((X + 2) ** 2 + 1)
    return {
        21: terme_a_terme(F[21], (1 - cos(2 * X)) / 2, LINEARISATION),
        22: parties(X**4, exp(X / 2), gestes=4),
        23: par_parties(X, sinh(X)),
        24: immediate(F[24], sp.tanh(X)),
        25: par_equation((sin(5 * X), exp(X / 2)), (10 * cos(5 * X), exp(X / 2))),
        26: par_equation((sin(ln(X)), sp.Integer(1)), (cos(ln(X)), sp.Integer(1))),
        27: terme_a_terme(F[27], (cosh(2 * X) - 1) / 2, LINEARISATION),
        28: terme_a_terme(F[28], carre, "decomposition"),
        29: changement_inverse(F[29], T**2, sqrt(X), 2 * T * exp(T)),
        30: par_parties(atan(X), sp.Integer(1)),
        31: par_parties(ln(A * X), X**P),
        32: terme_a_terme(F[32], X**2 / 2 + X**2 * cos(2 * X) / 2, LINEARISATION),
        33: par_parties(asin(X), sp.Integer(1)),
        34: changement(F[34], 1 / sqrt(U), 1 + exp(X)),
        35: changement(F[35], U ** R(27, 50) / 3, 1 + X**3),
        36: changement(F[36], 2 / (sqrt(3) * (1 + U**2)), (2 * X + 1) / sqrt(3)),
        37: parties((X + 1) ** 3, sin(X), gestes=3),
        38: changement_inverse(
            F[38], T**4 - 1, (X + 1) ** R(1, 4), 4 * (T + 1 + 1 / (T - 1))
        ),
        39: par_parties(atan(X), X**2 + 1),
        40: par_l_exponentielle_complexe(),
    }
