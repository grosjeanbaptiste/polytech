"""La methode de la surface, pour une equation implicite `F(x, y, y') = 0`.

En notant `z = y'`, `F(x, y, z) = 0` est une surface `S` de l'espace, sur
laquelle on cherche les lignes ou `dy = z dx`. Si `S` est parametree par
`x = a(u, v)`, `y = b(u, v)`, `z = c(u, v)`, cette condition devient

    (db/du - c da/du) du + (db/dv - c da/dv) dv = 0,

une equation differentielle entre les deux parametres.
"""

from __future__ import annotations

import sympy as sp


def forme_sur_la_surface(
    parametrage: tuple[object, object, object], u: sp.Symbol, v: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """Les facteurs de `du` et de `dv` dans `dy - z dx = 0`."""
    a, b, c = (sp.sympify(composante) for composante in parametrage)
    return (
        sp.simplify(sp.diff(b, u) - c * sp.diff(a, u)),
        sp.simplify(sp.diff(b, v) - c * sp.diff(a, v)),
    )
