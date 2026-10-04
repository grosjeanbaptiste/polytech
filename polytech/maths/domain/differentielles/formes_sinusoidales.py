"""Les formes dephasees d'une meme oscillation.

`A cos(wt) + B sin(wt)` s'ecrit aussi `K sin(wt + phi)` ou
`K cos(wt + psi)`, avec `K = sqrt(A^2 + B^2)` dans les deux cas :

* `A = K sin(phi)`, `B = K cos(phi)` ;
* `A = K cos(psi)`, `B = -K sin(psi)`.

Le syllabus ecrit `phi = arctg(A/B)`, qui ne vaut que pour `B > 0` :
l'arc tangente a deux arguments choisit le bon quadrant. Les deux
dephasages sont lies par `phi = psi + pi/2`, puisque
`sin(x) = cos(x - pi/2)`.
"""

from __future__ import annotations

import sympy as sp


def _amplitude(a: object, b: object) -> sp.Expr:
    amplitude = sp.sqrt(sp.sympify(a) ** 2 + sp.sympify(b) ** 2)
    if amplitude == 0:
        msg = "amplitude nulle : le dephasage n'est pas defini"
        raise ValueError(msg)
    return amplitude


def en_sinus(a: object, b: object) -> tuple[sp.Expr, sp.Expr]:
    """`(K, phi)` tels que `A cos(wt) + B sin(wt) = K sin(wt + phi)`."""
    return _amplitude(a, b), sp.atan2(sp.sympify(a), sp.sympify(b))


def en_cosinus(a: object, b: object) -> tuple[sp.Expr, sp.Expr]:
    """`(K, psi)` tels que `A cos(wt) + B sin(wt) = K cos(wt + psi)`."""
    return _amplitude(a, b), sp.atan2(-sp.sympify(b), sp.sympify(a))
