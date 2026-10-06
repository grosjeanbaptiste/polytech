"""Le residu d'un quotient `g / h`, par l'inverse d'une serie de puissances.

Si `g(a)` est non nul et que `h` a un zero d'ordre `m` en `a`,

    f = g / h = (z - a)^(-m) (somme a_k (z - a)^k) (somme b_(m+k) (z - a)^k)^(-1),

avec `a_k = g^(k)(a) / k!` et `b_k = h^(k)(a) / k!`. Si `d_k` sont les
coefficients de la serie inverse, donnes par la recurrence

    d_0 = 1 / b_m,    d_k = -(1 / b_m) somme_(p < k) d_p b_(m+k-p),

alors `Res(f, a) = somme_(p = 0)^(m-1) a_(m-1-p) d_p`. Pour `m = 1` c'est
`g(a) / h'(a)`.

Source : Mathematiques pour l'ingenieur III, syllabus, section 9.2,
methode 4, relations (113) et (114).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

ORDRE_MAXIMAL = 12
"""Au-dela, le zero est tenu pour non isole : la recherche s'arrete et le dit."""


def inverse_d_une_serie(coefficients: Sequence[object]) -> tuple[sp.Expr, ...]:
    """Les coefficients `c_k` de `C` telle que `B C = 1`, autant que ceux de `B`."""
    b = [sp.sympify(coefficient) for coefficient in coefficients]
    if b[0] == 0:
        msg = "serie de terme constant nul : elle n'a pas d'inverse"
        raise ValueError(msg)
    c: list[sp.Expr] = [1 / b[0]]
    for k in range(1, len(b)):
        somme = sum((c[p] * b[k - p] for p in range(k)), sp.Integer(0))
        c.append(sp.simplify(-somme / b[0]))
    return tuple(c)


def _coefficient(
    fonction: sp.Expr, variable: sp.Symbol, point: sp.Expr, k: int
) -> sp.Expr:
    """`f^(k)(a) / k!`, le coefficient de Taylor d'indice `k`."""
    derivee = sp.diff(fonction, variable, k).subs(variable, point)
    return sp.simplify(derivee / sp.factorial(k))


def ordre_du_zero(fonction: sp.Expr, variable: sp.Symbol, point: object) -> int:
    """Le rang de la premiere derivee non nulle en `a` ; 0 si `f(a)` est non nul."""
    a = sp.sympify(point)
    for ordre in range(ORDRE_MAXIMAL + 1):
        if _coefficient(sp.sympify(fonction), variable, a, ordre) != 0:
            return ordre
    msg = f"zero d'ordre superieur a {ORDRE_MAXIMAL} pour {fonction} en {point}"
    raise ValueError(msg)


def residu_d_un_quotient(
    numerateur: sp.Expr, denominateur: sp.Expr, variable: sp.Symbol, point: object
) -> sp.Expr:
    """`Res(g / h, a)` ; leve si `g(a)` est nul."""
    a = sp.sympify(point)
    g, h = sp.sympify(numerateur), sp.sympify(denominateur)
    if _coefficient(g, variable, a, 0) == 0:
        msg = f"le numerateur {g} s'annule en {a} : la methode ne s'applique pas"
        raise ValueError(msg)
    m = ordre_du_zero(h, variable, a)
    if m == 0:
        return sp.Integer(0)
    d = inverse_d_une_serie([_coefficient(h, variable, a, m + k) for k in range(m)])
    termes = (_coefficient(g, variable, a, m - 1 - p) * d[p] for p in range(m))
    return sp.simplify(sum(termes, sp.Integer(0)))
