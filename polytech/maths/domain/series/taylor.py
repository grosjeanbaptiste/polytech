"""Le developpement de Taylor d'ordre `k` en un point `a`, et son reste.

Section 4 de `Formulaire_Aide_memoire.pdf` (page 5) :

    P(x - a) = somme, pour n de 0 a k, de f^(n)(a) (x - a)^n / n!,
    R(x) = f^(k+1)(c) (x - a)^(k+1) / (k + 1)!,   c entre a et x.

En `a = 0` c'est le developpement de Mac Laurin (`mac_laurin.py`), dont
le formulaire donne quatre exemples jusqu'a leur terme general.

`f^(n)(a)` se prend par continuite quand la formule de `f` n'est pas
definie en `a` : c'est la valeur de la fonction prolongee, pourvu que la
limite existe et soit la meme des deux cotes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X

if TYPE_CHECKING:
    from collections.abc import Callable


def valeur_par_continuite(expression: sp.Expr, point: object) -> sp.Expr:
    """La valeur en `point`, ou la limite commune des deux cotes ; leve sinon."""
    valeur = expression.subs(X, point)
    if valeur.is_finite:
        return valeur
    gauche = sp.limit(expression, X, point, "-")
    droite = sp.limit(expression, X, point, "+")
    if gauche != droite or not gauche.is_finite:
        msg = (
            f"{expression} n'est pas derivable en {point} : limites {gauche}, {droite}"
        )
        raise ValueError(msg)
    return gauche


def coefficients_de_taylor(
    fonction: sp.Expr, point: object, nombre: int
) -> tuple[sp.Expr, ...]:
    """Les `nombre` premiers `f^(n)(a) / n!`."""
    return tuple(
        valeur_par_continuite(sp.diff(fonction, X, n), point) / sp.factorial(n)
        for n in range(nombre)
    )


def polynome_de_taylor(fonction: sp.Expr, point: object, ordre: int) -> sp.Expr:
    """`P_(f, k, a)`, ecrit en puissances de `x - a`."""
    if ordre < 0:
        msg = f"ordre negatif : {ordre}"
        raise ValueError(msg)
    coefficients = coefficients_de_taylor(fonction, point, ordre + 1)
    return sum(
        (c * (X - point) ** n for n, c in enumerate(coefficients)), sp.Integer(0)
    )


def reste_de_taylor(
    fonction: sp.Expr, point: object, ordre: int, intermediaire: sp.Expr
) -> sp.Expr:
    """`f^(k+1)(c) (x - a)^(k+1) / (k + 1)!`, pour un `c` entre `a` et `x`."""
    derivee = sp.diff(fonction, X, ordre + 1).subs(X, intermediaire)
    return derivee * (X - point) ** (ordre + 1) / sp.factorial(ordre + 1)


MACLAURIN_DU_FORMULAIRE: Final[dict[sp.Expr, Callable[[int], sp.Expr]]] = {
    sp.exp(X): lambda n: X**n / sp.factorial(n),
    sp.sin(X): lambda n: (-1) ** n * X ** (2 * n + 1) / sp.factorial(2 * n + 1),
    sp.cos(X): lambda n: (-1) ** n * X ** (2 * n) / sp.factorial(2 * n),
    1 / (1 + X): lambda n: (-1) ** n * X**n,
}
"""La fonction, et son terme de rang `n` tel que le formulaire l'ecrit."""
