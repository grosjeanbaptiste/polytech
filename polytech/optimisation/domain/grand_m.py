"""Chapitre II.5.2 : la constante `M`, « aussi grande que l'on veut ».

La methode des variables artificielles penalise celles-ci dans la
fonction economique par un cout `M` symbolique. Les coefficients `a_0j`
du tableau sont alors de la forme `a_0j^(1) + M a_0j^(2)`, que le cours
ecrit sur **deux lignes** (remarque 4 gamma, page 62).

Comparer deux tels coefficients, c'est comparer d'abord leur terme en
`M` : tant que la phase I n'est pas terminee, ce sont les `a_0j^(2)` qui
decident ; des qu'ils sont tous nuls, les `a_0j^(1)` reprennent la main.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

if TYPE_CHECKING:
    from polytech.optimisation.domain.tableau import TableauSimplexe

GRAND_M: Final = sp.Symbol("M", positive=True)
"""La constante symbolique de penalisation."""


def ordre(coefficient: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(terme en M, terme constant)` : l'ordre de `a + b M` pour `M` grand."""
    polynome = sp.Poly(sp.sympify(coefficient), GRAND_M)
    if polynome.degree() > 1:
        msg = f"coefficient qui n'est pas lineaire en M : {coefficient}"
        raise ValueError(msg)
    return polynome.coeff_monomial(GRAND_M), polynome.coeff_monomial(1)


def deux_lignes(tableau: TableauSimplexe) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """La ligne de `z` comme le cours l'ecrit : les termes en `M`, puis les autres."""
    ordres = [ordre(a) for a in tableau.coefficients[0, :]]
    return [o[0] for o in ordres], [o[1] for o in ordres]
