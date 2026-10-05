"""La transformee de Fourier a deux dimensions.

    F(w1, w2) = int int f(x1, x2) e^(-j (w1 x1 + w2 x2)) dx1 dx2.

C'est elle qui sert au traitement d'images : une image est un signal a
deux variables d'espace.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

PULSATIONS: Final = sp.symbols("omega_1 omega_2", real=True)
"""`w1` et `w2`, les deux variables de l'image."""


def transformee_a_deux_dimensions(
    fonction: sp.Expr, variables: tuple[sp.Symbol, sp.Symbol]
) -> sp.Expr:
    """`F(w1, w2)`, par deux integrations successives."""
    image = sp.sympify(fonction)
    for variable, pulsation in zip(variables, PULSATIONS, strict=True):
        noyau = sp.exp(-sp.I * pulsation * variable)
        image = sp.integrate(image * noyau, (variable, -sp.oo, sp.oo))
    return sp.simplify(image)
