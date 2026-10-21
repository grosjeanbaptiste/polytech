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


def integrer_sur_la_surface(
    parametrage: tuple[object, object, object], u: sp.Symbol, v: sp.Symbol
) -> sp.Expr:
    """`G(u, v)`, constante le long des solutions, quand `A du + B dv = 0` se separe.

    `dv / du = -A / B = f(u) g(v)` donne `int dv / g(v) - int f(u) du = C`
    (exemple 1.18, page 33).
    """
    en_u, en_v = forme_sur_la_surface(parametrage, u, v)
    pente = sp.simplify(-en_u / en_v)
    facteurs = sp.separatevars(pente, symbols=[u, v], dict=True)
    if facteurs is None:
        msg = f"dv/du = {pente} ne se separe pas en f(u) g(v)"
        raise ValueError(msg)
    f = facteurs["coeff"] * facteurs[u]
    return sp.simplify(sp.integrate(1 / facteurs[v], v) - sp.integrate(f, u))
