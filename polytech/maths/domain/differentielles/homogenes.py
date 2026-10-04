"""Les equations homogenes de degre 0, `y' = f(y / x)` (planches 36-37).

`f(x, y)` est **homogene de degre `n`** si `f(l x, l y) = l^n f(x, y)`
pour tout `l`. Au degre 0, `f` ne depend que du rapport `y / x`, et le
changement d'inconnue `z = y / x` -- `y = z x`, `y' = z + x z'` -- donne

    x z' = f(z) - z,

qui est a variables separees.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE

Z: Final = sp.Symbol("z", real=True)
"""Le rapport `y / x`, nouvelle inconnue."""

FACTEUR: Final = sp.Symbol("ell", positive=True)


def degre_d_homogeneite(fonction: sp.Expr) -> sp.Expr | None:
    """`n` tel que `f(l x, l y) = l^n f(x, y)`, ou `None` si `f` n'est pas homogene."""
    dilatee = fonction.subs(
        {VARIABLE: FACTEUR * VARIABLE, INCONNUE: FACTEUR * INCONNUE}, simultaneous=True
    )
    rapport = sp.simplify(dilatee / fonction)
    if rapport.has(VARIABLE) or rapport.has(INCONNUE):
        return None
    return sp.simplify(sp.log(rapport) / sp.log(FACTEUR))


def equation_en_z(second_membre: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(f(z), f(z) - z)` : le second membre en `z`, et ce que vaut `x z'`."""
    if degre_d_homogeneite(second_membre) != 0:
        msg = f"second membre pas homogene de degre 0 : {second_membre}"
        raise ValueError(msg)
    en_z = sp.simplify(second_membre.subs(INCONNUE, Z * VARIABLE))
    return en_z, sp.simplify(en_z - Z)
