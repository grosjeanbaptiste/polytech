"""La transformee de Laplace comme transformee integrale, et d'ou elle vient.

Une **transformee integrale** associe a `y(t)` l'integrale de `y` contre
un noyau `K(p, t)`. Celle de Laplace prend `K(p, t) = e^(-pt)` sur
`[0, +oo[` :

    X(p) = int_0^oo e^(-pt) x(t) dt,     p = s + j omega.

C'est une integrale impropre : la limite, quand `beta` croit, de
l'integrale tronquee a `beta`. Elle ne voit que `t >= 0` (causalite).

**Pourquoi cette forme ?** Laplace partait des series entieres
`A(x) = sum a(n) x^n`. En passant d'un indice entier `n` a une variable
continue `t`, la somme devient `int_0^oo a(t) x^t dt` ; et comme `x^t` se
derive mal, on pose `x = e^(-s)` : `x^t = e^(-st)`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import IMAGE, TEMPS

NOYAU: Final = sp.exp(-IMAGE * TEMPS)
"""`K(p, t) = e^(-pt)`."""

_S: Final = sp.Symbol("sigma", positive=True)


def serie_generatrice(
    terme: sp.Expr, indice: sp.Symbol, variable: sp.Symbol
) -> sp.Expr:
    """`A(x) = sum_{n >= 0} a(n) x^n`."""
    return sp.summation(terme * variable**indice, (indice, 0, sp.oo))


def version_continue_de_la_serie(terme: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """`int_0^oo a(t) x^t dt`, pour `0 < x < 1`, calculee avec `x = e^(-s)`."""
    en_s = sp.integrate(terme * sp.exp(-_S * TEMPS), (TEMPS, 0, sp.oo))
    return sp.simplify(en_s.subs(_S, -sp.log(variable)))


def integrale_tronquee(original: sp.Expr, borne: sp.Expr) -> sp.Expr:
    """`int_0^beta e^(-pt) x(t) dt`, avant le passage a la limite."""
    return sp.simplify(sp.integrate(NOYAU * original, (TEMPS, 0, borne)))


def transformee_par_l_integrale(original: sp.Expr) -> sp.Expr:
    """`X(p)` comme limite de l'integrale tronquee, pour `p > 0`."""
    borne = sp.Symbol("beta", positive=True)
    limite = sp.limit(integrale_tronquee(original, borne), borne, sp.oo)
    if not limite.is_finite:
        msg = f"l'integrale de Laplace de {original} ne converge pas pour tout p > 0"
        raise ValueError(msg)
    return sp.simplify(limite)
