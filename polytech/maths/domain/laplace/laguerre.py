"""Les polynomes de Laguerre, par la transformee de Laplace.

L'equation de Laguerre `t y'' + (1 - t) y' + n y = 0` a des coefficients
qui dependent de `t`. Son image est une equation du **premier** ordre en
`Y(p)`, a variables separees :

    dY / Y = (n + 1 - p) / ((p - 1) p) dp,     Y(p) = (p - 1)^n / p^(n+1).

Son original est le polynome de Laguerre de degre `n`,

    L_n(t) = e^t / n! . d^n/dt^n (t^n e^(-t)).
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import IMAGE, TEMPS


def image_de_laguerre(degre: int) -> sp.Expr:
    """`(p - 1)^n / p^(n+1)`."""
    return (IMAGE - 1) ** degre / IMAGE ** (degre + 1)


def polynome_de_laguerre(degre: int) -> sp.Expr:
    """`L_n(t)`, par la formule de Rodrigues."""
    if degre < 0:
        msg = f"degre negatif : {degre}"
        raise ValueError(msg)
    derivee = sp.diff(TEMPS**degre * sp.exp(-TEMPS), TEMPS, degre)
    return sp.expand(sp.simplify(sp.exp(TEMPS) * derivee / sp.factorial(degre)))
