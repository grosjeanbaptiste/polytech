"""Le produit de convolution de deux fonctions causales.

    (f * g)(t) = int_0^t f(t - s) g(s) ds.

Il est commutatif, associatif et distributif sur la somme, et la
transformee de Laplace le change en produit ordinaire :
`L(f * g) = L(f) L(g)`. La fonction 1 n'en est **pas** l'element neutre
(`1 * e^t = e^t - 1`) ; le neutre est l'impulsion de Dirac, qui n'est pas
une fonction mais une mesure -- `int f d(delta) = f(0)` -- et dont l'image
est 1.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import TEMPS

IMPULSION_DE_DIRAC: Final = sp.DiracDelta(TEMPS)
"""`delta`, notee comme une fonction de `t` par abus tolere."""

_MUETTE: Final = sp.Symbol("s", real=True, nonnegative=True)


def produit_de_convolution(premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
    """`int_0^t f(t - s) g(s) ds` ; l'impulsion de Dirac y est neutre."""
    if premiere == IMPULSION_DE_DIRAC:
        return seconde
    if seconde == IMPULSION_DE_DIRAC:
        return premiere
    integrande = premiere.subs(TEMPS, TEMPS - _MUETTE) * seconde.subs(TEMPS, _MUETTE)
    return sp.simplify(sp.integrate(integrande, (_MUETTE, 0, TEMPS)))
