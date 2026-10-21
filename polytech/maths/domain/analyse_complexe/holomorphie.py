"""L'holomorphie, lue sur les parties reelle et imaginaire.

`f = u + i v` est holomorphe sur un ouvert si et seulement si `u` et `v` y
sont de classe C1 et verifient les relations de Cauchy-Riemann,
`u_x = v_y` et `u_y = -v_x` (proposition 9). La derivee vaut alors
`f' = d_x f = u_x + i v_x` (proposition 10), et `u`, `v` sont harmoniques
(proposition 11).

Les fonctions traitees sont des expressions sympy elementaires, de classe
C1 la ou elles sont definies : c'est sur cet ouvert que le predicat se
prononce. Les relations doivent y valoir identiquement -- une egalite en un
seul point ne fait pas une fonction holomorphe.

Source : Mathematiques pour l'ingenieur III, syllabus, section 4.1, page 6.
"""

from __future__ import annotations

import sympy as sp

X = sp.Symbol("x", real=True)
Y = sp.Symbol("y", real=True)
Z = sp.Symbol("z")
"""La variable complexe ; `z = x + i y`."""


def parties_reelle_et_imaginaire(fonction: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(u, v)` tels que `f(x + i y) = u(x, y) + i v(x, y)`."""
    valeur = sp.expand_complex(sp.sympify(fonction).subs(Z, X + sp.I * Y))
    return sp.simplify(sp.re(valeur)), sp.simplify(sp.im(valeur))


def _nulle(expression: sp.Expr) -> bool:
    return bool(sp.simplify(expression) == 0)


def verifie_cauchy_riemann(u: sp.Expr, v: sp.Expr) -> bool:
    """`u_x = v_y` et `u_y = -v_x`, identiquement en `x` et `y`."""
    premiere = sp.diff(u, X) - sp.diff(v, Y)
    seconde = sp.diff(u, Y) + sp.diff(v, X)
    return _nulle(premiere) and _nulle(seconde)


def est_holomorphe(fonction: sp.Expr) -> bool:
    """La fonction de `Z` verifie Cauchy-Riemann sur son ouvert de definition."""
    return verifie_cauchy_riemann(*parties_reelle_et_imaginaire(fonction))


def derivee_complexe(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """`f' = d_x f = u_x + i v_x` (proposition 10)."""
    return sp.diff(u, X) + sp.I * sp.diff(v, X)


def sont_harmoniques(u: sp.Expr, v: sp.Expr) -> bool:
    """`u_xx + u_yy = 0` et `v_xx + v_yy = 0` (proposition 11)."""
    return all(_nulle(sp.diff(w, X, 2) + sp.diff(w, Y, 2)) for w in (u, v))
