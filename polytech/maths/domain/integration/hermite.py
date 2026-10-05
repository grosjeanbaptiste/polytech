"""Les fractions rationnelles a racines multiples, sans recurrence.

Quand le denominateur `Q` a des facteurs repetes, on ecrit

    P / Q = (A / Q_1)' + B / Q_2,

ou `Q_1` porte chaque facteur de `Q` avec un exposant diminue de un, et
`Q_2 = Q / Q_1` chaque facteur une seule fois ; `A` et `B` sont de degre
moindre que `Q_1` et `Q_2`. La partie `A / Q_1` sort toute primitivee, et
il ne reste qu'une fraction a racines simples.

Pour une racine **simple** `a`, le coefficient de `1/(x - a)` se lit
directement : c'est `G(a)`, ou `G = (x - a) P / Q`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


def decomposition_d_hermite(
    numerateur: object, denominateur: object
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """`(A, Q_1, B, Q_2)` tels que `P/Q = (A/Q_1)' + B/Q_2`."""
    p, q = sp.sympify(numerateur), sp.sympify(denominateur)
    q1 = sp.gcd(q, sp.diff(q, VARIABLE))
    q2 = sp.cancel(q / q1)
    degres = [int(sp.degree(polynome, VARIABLE)) for polynome in (q1, q2)]
    inconnues = [
        sp.symbols(f"{nom}0:{max(d, 1)}") for nom, d in zip("ab", degres, strict=True)
    ]
    a, b = (
        sum((c * VARIABLE**k for k, c in enumerate(coefficients[:d])), sp.Integer(0))
        for coefficients, d in zip(inconnues, degres, strict=True)
    )
    derivee = sp.cancel(sp.diff(q1, VARIABLE) * q2 / q1)
    identite = sp.expand(sp.diff(a, VARIABLE) * q2 - a * derivee + b * q1 - p)
    equations = sp.Poly(identite, VARIABLE).all_coeffs()
    (valeurs,) = sp.solve(equations, [*inconnues[0], *inconnues[1]], dict=True) or [{}]
    libres = dict.fromkeys([*inconnues[0], *inconnues[1]], 0)
    return tuple(sp.expand(e.subs(valeurs).subs(libres)) for e in (a, q1, b, q2))


def contribution_d_une_racine(fraction: sp.Expr, racine: object) -> sp.Expr:
    """`G(a) = lim (x - a) P/Q`, le coefficient de `1/(x - a)`.

    Leve si `a` n'est pas une racine simple du denominateur.
    """
    a = sp.sympify(racine)
    valeur = sp.limit(sp.cancel((VARIABLE - a) * fraction), VARIABLE, a)
    if not valeur.is_finite or valeur == 0:
        msg = f"{a} n'est pas une racine simple du denominateur"
        raise ValueError(msg)
    return sp.simplify(valeur)


def inverse_d_une_puissance(exposant: int) -> sp.Expr:
    """`I_n = int dx / (x^2 + 1)^n`, par la recurrence sur `n`."""
    primitive = sp.atan(VARIABLE)
    for n in range(2, exposant + 1):
        terme = VARIABLE / ((2 * n - 2) * (VARIABLE**2 + 1) ** (n - 1))
        primitive = sp.Rational(2 * n - 3, 2 * n - 2) * primitive + terme
    return primitive
