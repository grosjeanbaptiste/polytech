"""Recueil de primitives, exercice 3 : la recurrence sur `1 / (x^2 + 1)^n`.

Une integration par parties de `I_(n-1)`, avec `u = (x^2 + 1)^(1-n)` et
`dv = dx`, fait apparaitre `I_n` : c'est la relation que l'enonce demande
de demontrer. Le trinome `x^2 + x + 1` s'y ramene en completant le carre.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.integration.hermite import inverse_d_une_puissance
from polytech.maths.domain.symboles import ENTIER as N
from polytech.maths.domain.symboles import VARIABLE as X

U = sp.Symbol("u", real=True)
TRINOME = sp.Symbol("T", positive=True)
"""`u` et `T = x^2 + x + 1`, laisses en lettres dans la forme que le syllabus ecrit."""
CENTRE = (2 * X + 1) / sp.sqrt(3)
"""`u` tel que `x^2 + x + 1 = 3/4 (u^2 + 1)`."""


def relation() -> tuple[sp.Expr, sp.Expr]:
    """`I_n = facteur . I_(n-1) + terme` : le facteur, puis le terme integre."""
    return (2 * N - 3) / (2 * N - 2), X / ((2 * N - 2) * (X**2 + 1) ** (N - 1))


def puissance(ordre: int) -> sp.Expr:
    """`I_n`, la primitive de `1 / (x^2 + 1)^n` ; leve pour un ordre non positif."""
    if ordre < 1:
        msg = f"l'ordre doit etre un entier strictement positif : {ordre}"
        raise ValueError(msg)
    return inverse_d_une_puissance(ordre)


def facteur_du_trinome(ordre: int) -> sp.Expr:
    """`(4/3)^n . racine(3)/2` : ce que le changement `u` met devant `I_n(u)`."""
    return sp.Rational(4, 3) ** ordre * sp.sqrt(3) / 2


def sur_le_trinome(ordre: int) -> sp.Expr:
    """La primitive de `1 / (x^2 + x + 1)^n`, par `u = (2x + 1) / racine(3)`."""
    return facteur_du_trinome(ordre) * puissance(ordre).subs(X, CENTRE)


def coefficients_du_trinome(ordre: int) -> tuple[sp.Expr, ...]:
    """Les coefficients de `arctg u`, `u / T`, `u / T^2`, ... avec `T = x^2 + x + 1`.

    C'est l'ecriture du syllabus. Dans `I_n(u)`, `1 / (u^2 + 1)^k` vaut
    `(3/4)^k / T^k` : chaque terme de la recurrence prend ce facteur.
    """
    en_u = [sp.Integer(1)]
    for n in range(2, ordre + 1):
        en_u = [sp.Rational(2 * n - 3, 2 * n - 2) * c for c in en_u]
        en_u.append(sp.Rational(1, 2 * n - 2))
    return tuple(c * sp.Rational(3, 4) ** k for k, c in enumerate(en_u))


def forme_du_trinome(ordre: int) -> sp.Expr:
    """La primitive de `1 / T^n`, ecrite en `u` et `T` comme le syllabus."""
    coefficients = coefficients_du_trinome(ordre)
    termes = [c * U / TRINOME**k for k, c in enumerate(coefficients) if k]
    interieur = sp.Add(coefficients[0] * sp.atan(U), *termes)
    return sp.Mul(facteur_du_trinome(ordre), interieur, evaluate=False)
