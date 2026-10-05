"""Sommer une serie alternee a une precision donnee.

Quand les termes changent de signe a chaque rang et que leur module tend
vers 0 en decroissant, le reste `R_N` est majore par le premier terme
neglige : `|R_N| <= |u_(N+1)|`. On somme donc jusqu'au rang `N` ou ce
terme passe sous la tolerance.
"""

from __future__ import annotations

import sympy as sp


def somme_alternee(
    terme: sp.Expr, indice: sp.Symbol, tolerance: object
) -> tuple[int, sp.Expr, sp.Expr]:
    """`(N, S_N, |u_(N+1)|)`, au premier rang ou `|u_(N+1)|` passe sous la tolerance.

    La somme part du rang 0. Leve si la serie n'est pas alternee.
    """
    premier, second = terme.subs(indice, 0), terme.subs(indice, 1)
    if premier * second >= 0:
        msg = f"serie non alternee : {premier} puis {second} sont de meme signe"
        raise ValueError(msg)
    rang, somme = 0, premier
    while sp.Abs(terme.subs(indice, rang + 1)) >= tolerance:
        rang += 1
        somme += terme.subs(indice, rang)
    return rang, somme, sp.Abs(terme.subs(indice, rang + 1))
