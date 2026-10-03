"""La transformation triangle-etoile, ou theoreme de Kennelly.

Trois resistances en triangle entre les noeuds `a`, `b`, `c` se
remplacent par trois resistances en etoile, reliees a un noeud central :
chaque branche vaut le produit des deux cotes qui touchent son sommet,
divise par la somme des trois.

Elle sert a reduire un **pont**, que les associations serie et parallele
ne savent pas reduire. La verifier, c'est verifier ce qu'elle conserve :
la resistance vue entre deux sommets quand le troisieme est en l'air.
"""

from __future__ import annotations

import sympy as sp


def triangle_vers_etoile(
    entre_a_et_b: sp.Expr, entre_b_et_c: sp.Expr, entre_c_et_a: sp.Expr
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Les branches de l'etoile issues de `a`, `b` et `c` ; leve sur un cote nul."""
    cotes = [sp.sympify(c) for c in (entre_a_et_b, entre_b_et_c, entre_c_et_a)]
    if any(c == 0 for c in cotes):
        msg = f"resistance nulle dans le triangle : {cotes} -- deux sommets confondus"
        raise ValueError(msg)
    ab, bc, ca = cotes
    somme = ab + bc + ca
    return (
        sp.simplify(ab * ca / somme),
        sp.simplify(ab * bc / somme),
        sp.simplify(bc * ca / somme),
    )
