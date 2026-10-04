"""La transformee de Laplace devant un coefficient qui depend de `t`.

Multiplier l'original par `t` revient a deriver l'image :

    L[t y]   = -Y'(p),            L[t^n y] = (-1)^n Y^(n)(p),
    L[t y']  = -Y - p Y',
    L[t y''] = -2 p Y - p^2 Y' + y(0).

Une equation a coefficients polynomiaux en `t` ne devient donc plus une
equation algebrique, mais une equation differentielle en `Y(p)` -- d'ordre
moindre, et c'est le gain.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import IMAGE


def du_produit_par_t(image: sp.Expr, puissance: int = 1) -> sp.Expr:
    """`L[t^n y] = (-1)^n Y^(n)(p)`."""
    return (-1) ** puissance * sp.diff(image, IMAGE, puissance)


def de_t_fois_la_derivee(image: sp.Expr) -> sp.Expr:
    """`L[t y'] = -Y - p Y'` : la derivee de `p Y - y(0)`, changee de signe."""
    return -image - IMAGE * sp.diff(image, IMAGE)


def de_t_fois_la_derivee_seconde(image: sp.Expr, initiale: object) -> sp.Expr:
    """`L[t y''] = -2 p Y - p^2 Y' + y(0)`."""
    return -2 * IMAGE * image - IMAGE**2 * sp.diff(image, IMAGE) + sp.sympify(initiale)
