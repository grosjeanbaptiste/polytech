"""Les proprietes de l'integrale definie (planche 32).

* **linearite** : `int (c f + d g) = c int f + d int g` ;
* **bornes** : `int_a^b = int_a^c + int_c^b` pour tout `c` ;
  `int_a^b = -int_b^a` ; `int_a^a = 0`.

`IntegraleDefinie` refuse des bornes inversees -- dans un corrige, c'est
presque toujours une faute de recopie. Les proprietes ci-dessus demandent
pourtant l'integrale **orientee** : elle vit ici, et s'appuie sur
`IntegraleDefinie` une fois les bornes remises dans l'ordre.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.integration.definie import IntegraleDefinie


def integrale_orientee(integrande: sp.Expr, depart: object, arrivee: object) -> sp.Expr:
    """`int_depart^arrivee f(x) dx`, quel que soit l'ordre des bornes."""
    a, b = sp.sympify(depart), sp.sympify(arrivee)
    if a == b:
        return sp.Integer(0)
    if a < b:
        return IntegraleDefinie(integrande, a, b).valeur()
    return -IntegraleDefinie(integrande, b, a).valeur()
