"""Resoudre une equation differentielle lineaire par une serie entiere.

Pour des coefficients qui dependent de `x`, ni l'equation caracteristique
ni la transformee de Laplace ne suffisent. On pose `y = somme a_n x^n`,
on derive terme a terme, on substitue, et l'on annule le coefficient de
chaque puissance de `x` : les `a_n` se deduisent, de proche en proche, des
premiers -- autant de constantes libres que l'ordre de l'equation.

Le developpement se fait autour d'un **point ordinaire**, ou le
coefficient de la derivee la plus haute ne s'annule pas.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

INCONNUES: Final = sp.symbols("a0 a1")
"""Les constantes libres `a_0` et `a_1` (seule `a_0` au premier ordre)."""


def solution_en_serie(coefficients: tuple[object, ...], ordre: int) -> sp.Expr:
    """La solution de `somme p_k(x) y^(k) = 0`, tronquee a `x^ordre`.

    `coefficients` donne `p_0, p_1, ..., p_m`, du coefficient de `y` a celui
    de la derivee la plus haute.
    """
    polynomes = [sp.sympify(p) for p in coefficients]
    degre = len(polynomes) - 1
    if polynomes[-1].subs(VARIABLE, 0) == 0:
        msg = "point singulier en 0 : le coefficient de la plus haute derivee s'annule"
        raise ValueError(msg)
    a = [*INCONNUES[:degre], *sp.symbols(f"c{degre}:{ordre + 1}")]
    serie = sum((a[n] * VARIABLE**n for n in range(ordre + 1)), sp.Integer(0))
    residu = sp.expand(
        sum(p * sp.diff(serie, VARIABLE, k) for k, p in enumerate(polynomes))
    )
    conditions = [residu.coeff(VARIABLE, j) for j in range(ordre - degre + 1)]
    (valeurs,) = sp.solve(conditions, a[degre:], dict=True)
    return sp.expand(serie.subs(valeurs))
