"""La serie de Mac Laurin d'une fonction, et ce qui la separe de la fonction.

Les coefficients d'une serie entiere sont ceux du developpement de Mac
Laurin de sa somme : `a_n = S^(n)(0) / n!`. La reciproque n'est pas
gratuite : une fonction indefiniment derivable n'egale sa serie de Mac
Laurin que si le reste de Lagrange

    R_n = f^(n+1)(theta x) x^(n+1) / (n + 1)!,     0 < theta < 1,

tend vers 0. Ce n'est pas toujours le cas.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def coefficients_de_mac_laurin(fonction: sp.Expr, nombre: int) -> tuple[sp.Expr, ...]:
    """Les `nombre` premiers `f^(n)(0) / n!`."""
    return tuple(
        sp.diff(fonction, VARIABLE, n).subs(VARIABLE, 0) / sp.factorial(n)
        for n in range(nombre)
    )


def reste_de_lagrange(fonction: sp.Expr, ordre: int, theta: sp.Expr) -> sp.Expr:
    """`f^(n+1)(theta x) x^(n+1) / (n + 1)!`."""
    derivee = sp.diff(fonction, VARIABLE, ordre + 1).subs(VARIABLE, theta * VARIABLE)
    return derivee * VARIABLE ** (ordre + 1) / sp.factorial(ordre + 1)
