"""La nature d'une singularite isolee : reguliere, pole d'ordre `h`, essentielle.

Autour d'une singularite isolee `a`, une fonction holomorphe s'ecrit
`somme a_k (z - a)^k`, `k` de `-oo` a `+oo`. Trois cas, et trois seulement :

- aucune puissance negative : la singularite est **reguliere** ;
- un nombre fini, la plus basse etant `(z - a)^(-h)` : `a` est un **pole
  d'ordre h**, et `f(z) (z - a)^h` a une limite finie et non nulle ;
- une infinite : la singularite est **essentielle**.

Source : Mathematiques pour l'ingenieur III, syllabus, section 7.4.
"""

from __future__ import annotations

import sympy as sp
from sympy.core.function import PoleError

ECART = sp.Symbol("u")
"""`z - a`, l'ecart a la singularite."""


def autour_de(fonction: sp.Expr, variable: sp.Symbol, point: object) -> sp.Expr:
    """La fonction ecrite en `u = z - a`."""
    return sp.sympify(fonction).subs(variable, sp.sympify(point) + ECART)


def ordre_du_pole(fonction: sp.Expr, variable: sp.Symbol, point: object) -> int:
    """`h` tel que `f (z - a)^h` ait une limite finie non nulle ; 0 si reguliere.

    Leve sur une singularite essentielle : aucun `h` ne convient.
    """
    locale = autour_de(fonction, variable, point)
    try:
        coefficient, exposant = locale.leadterm(ECART)
    except (PoleError, NotImplementedError, ValueError) as erreur:
        msg = f"singularite essentielle de {fonction} en {point} : aucun ordre"
        raise ValueError(msg) from erreur
    if coefficient.has(sp.AccumBounds) or not exposant.is_integer:
        msg = f"singularite essentielle de {fonction} en {point} : aucun ordre"
        raise ValueError(msg)
    return max(0, -int(exposant))
