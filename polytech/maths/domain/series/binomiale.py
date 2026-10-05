"""La serie binomiale : `(1 + x)^a` pour un exposant reel quelconque.

    (1 + x)^a = 1 + a x + a(a-1)/2! x^2 + ... + a(a-1)...(a-n+1)/n! x^n + ...

pour `|x| < 1`.

Le quotient de deux coefficients successifs vaut `(a - n) / (n + 1)`, de
module tendant vers 1 : le rayon de convergence est 1. Que la somme soit
bien `(1 + x)^a` se montre par une voie indirecte : elle verifie
`(1 + x) y' = a y`, `y(0) = 1`, equation dont les coefficients obeissent a
`n b_n = (a - n + 1) b_(n-1)` et que seule cette serie satisfait.

Pour `a` entier naturel la serie s'arrete : c'est la formule du binome.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.series.entieres import INDICE
from polytech.maths.domain.symboles import VARIABLE


def coefficient_binomial(exposant: object, rang: int) -> sp.Expr:
    """`a (a - 1) ... (a - n + 1) / n!`."""
    a = sp.sympify(exposant)
    produit = sp.Mul(*(a - k for k in range(rang)))
    return sp.expand(produit / sp.factorial(rang))


def developpement_binomial(exposant: object, ordre: int) -> sp.Expr:
    """La somme partielle de la serie binomiale, jusqu'a `x^ordre`."""
    return sum(
        (coefficient_binomial(exposant, n) * VARIABLE**n for n in range(ordre + 1)),
        sp.Integer(0),
    )


def quotient_binomial(exposant: object) -> sp.Expr:
    """`a_(n+1) / a_n = (a - n) / (n + 1)`."""
    return (sp.sympify(exposant) - INDICE) / (INDICE + 1)
