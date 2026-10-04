"""La limite d'une fraction rationnelle quand un parametre tend vers l'infini.

Les circuits du cours se resolvent en fractions rationnelles de leurs
composants : une resistance, une transconductance ou un gain d'AOP qui
tend vers l'infini se traite alors exactement, sans `sympy.limit`, qui
s'enlise des que les symboles se multiplient.
"""

from __future__ import annotations

import sympy as sp

_EPSILON = sp.Symbol("epsilon", positive=True)


def limite_a_l_infini(expression: sp.Expr, symbole: sp.Symbol) -> sp.Expr:
    """La limite d'une fraction rationnelle quand `symbole -> oo`.

    Avec `symbole = 1/epsilon`, la limite se lit sur les termes de plus bas
    degre en `epsilon` du numerateur et du denominateur.
    """
    numerateur, denominateur = sp.fraction(
        sp.cancel(expression.subs(symbole, 1 / _EPSILON))
    )
    haut = sp.Poly(numerateur, _EPSILON)
    bas = sp.Poly(denominateur, _EPSILON)
    ordre_haut = min(m[0] for m in haut.monoms())
    ordre_bas = min(m[0] for m in bas.monoms())
    if ordre_haut < ordre_bas:
        msg = f"limite infinie quand {symbole} -> oo : {expression}"
        raise ValueError(msg)
    if ordre_haut > ordre_bas:
        return sp.Integer(0)
    return sp.cancel(
        haut.coeff_monomial(_EPSILON**ordre_haut)
        / bas.coeff_monomial(_EPSILON**ordre_bas)
    )
