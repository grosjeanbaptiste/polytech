"""Le produit de Cauchy de deux series entieres.

Le produit de `somme u_n x^n` par `somme v_n x^n` est la serie de terme
general `c_n x^n`, ou

    c_n = u_0 v_n + u_1 v_(n-1) + ... + u_n v_0

rassemble les produits dont les rangs ont pour somme `n`. Si les deux
series convergent absolument, la serie produit converge vers le produit
des sommes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.maths.domain.series.entieres import SerieEntiere


def produit_de_cauchy(
    premiere: SerieEntiere, seconde: SerieEntiere, ordre: int
) -> tuple[sp.Expr, ...]:
    """Les coefficients `c_0, ..., c_ordre` de la serie produit."""
    return tuple(
        sp.simplify(
            sum(
                (
                    premiere.coefficient(k) * seconde.coefficient(n - k)
                    for k in range(n + 1)
                ),
                sp.Integer(0),
            )
        )
        for n in range(ordre + 1)
    )
