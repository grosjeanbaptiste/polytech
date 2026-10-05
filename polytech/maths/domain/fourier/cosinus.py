"""La transformee de Fourier en cosinus, pour une fonction donnee sur `[0, +oo[`.

    F_c(lambda) = int_0^oo f(t) cos(lambda t) dt,
    f(x) = (2 / pi) int_0^oo F_c(lambda) cos(lambda x) dlambda.

C'est la transformee de Fourier du prolongement pair de `f` : l'analogue,
sur un domaine non borne, de la serie de cosinus.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def transformee_en_cosinus(fonction: sp.Expr, valeur: sp.Symbol) -> sp.Expr:
    """`int_0^oo f(x) cos(lambda x) dx`."""
    integrande = fonction * sp.cos(valeur * VARIABLE)
    return sp.simplify(sp.integrate(integrande, (VARIABLE, 0, sp.oo)))
