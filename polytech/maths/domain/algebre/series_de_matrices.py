"""Les suites et les series de matrices carrees.

Une suite `A_n` converge vers `A` si `d(A_n, A)` tend vers 0, c'est-a-dire
si chaque coefficient converge. Une serie converge si la suite de ses
sommes partielles converge ; il suffit que la serie numerique des normes
converge.

La **serie geometrique** `I + A + A^2 + ...` est majoree en norme par
`somme ||A||^n` : elle converge des que `||A|| < 1`, et sa somme est
`(I - A)^-1`. Cette condition est suffisante, non necessaire : ce qui
decide, c'est que toutes les valeurs propres soient de module inferieur
a 1.
"""

from __future__ import annotations

import sympy as sp


def _limite(element: sp.Expr, indice: sp.Symbol) -> sp.Expr:
    """La limite d'un coefficient ; une somme jusqu'a `n` devient une serie."""
    serie = element.replace(
        lambda e: isinstance(e, sp.Sum) and e.limits[0][2] == indice,
        lambda e: sp.Sum(e.function, (e.limits[0][0], e.limits[0][1], sp.oo)),
    )
    return sp.limit(serie.doit(), indice, sp.oo)


def limite_de_la_suite(suite: sp.Matrix, indice: sp.Symbol) -> sp.Matrix:
    """La matrice des limites ; leve si un coefficient n'en a pas de finie."""
    limite = suite.applyfunc(lambda e: _limite(e, indice))
    if not all(e.is_finite for e in limite):
        msg = f"suite de matrices divergente : {limite.tolist()}"
        raise ValueError(msg)
    return limite


def somme_partielle_geometrique(matrice: sp.Matrix, ordre: int) -> sp.Matrix:
    """`I + A + ... + A^ordre`."""
    somme, puissance = sp.eye(matrice.rows), sp.eye(matrice.rows)
    for _ in range(ordre):
        puissance = puissance * matrice
        somme += puissance
    return somme.applyfunc(sp.expand)


def somme_geometrique(matrice: sp.Matrix) -> sp.Matrix:
    """`(I - A)^-1` ; leve si une valeur propre est, a coup sur, de module >= 1.

    Devant un parametre, rien ne se decide : la somme rendue ne vaut alors
    que la ou les valeurs propres restent de module inferieur a 1.
    """
    if any(sp.Ge(sp.Abs(valeur), 1) == sp.true for valeur in matrice.eigenvals()):
        msg = "serie geometrique divergente : une valeur propre de module >= 1"
        raise ValueError(msg)
    return sp.simplify((sp.eye(matrice.rows) - matrice).inv())
