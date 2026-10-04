"""Les transformations contractantes et leur point fixe.

`T` est **contractante** si `d(T(x), T(y)) <= K d(x, y)` avec `0 <= K < 1`.
Dans un espace metrique complet elle a alors un point fixe `u = T(u)`, et
un seul ; toute suite `x_n = T(x_(n-1))` y converge, d'ou qu'elle parte,
et d'autant plus vite que `K` est petit :

    d(x_n, u) <= K^n / (1 - K) . d(x_0, x_1).

Pour une fonction derivable de la droite reelle, il suffit que
`|T'(x)| <= K < 1` (theoreme des accroissements finis).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.series.uniforme import distance_uniforme

if TYPE_CHECKING:
    from collections.abc import Callable


def iterer[T](
    transformation: Callable[[T], T], depart: T, nombre: int
) -> tuple[T, ...]:
    """`x_0, T(x_0), T(T(x_0)), ...`, jusqu'a `x_nombre`."""
    suite = [depart]
    for _ in range(nombre):
        suite.append(transformation(suite[-1]))
    return tuple(suite)


def constante_de_contraction(
    transformation: sp.Expr, variable: sp.Symbol, intervalle: sp.Set
) -> sp.Expr:
    """`sup |T'|` sur l'intervalle : la plus petite constante `K` qui convienne."""
    derivee = sp.diff(transformation, variable)
    return distance_uniforme(derivee, 0, variable, intervalle)


def borne_de_l_erreur(constante: object, premier_pas: object, rang: int) -> sp.Expr:
    """`K^n / (1 - K) . d(x_0, x_1)`, qui majore `d(x_n, u)`."""
    k = sp.sympify(constante)
    if not 0 <= k < 1:
        msg = f"transformation non contractante : K = {k} n'est pas dans [0, 1["
        raise ValueError(msg)
    return k**rang / (1 - k) * sp.sympify(premier_pas)
