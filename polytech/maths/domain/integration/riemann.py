"""L'integrale definie comme limite de sommes de rectangles (planche 30).

    int_a^b f(x) dx = lim sum f(xi_i) dx_i   quand dx -> 0.

Le decoupage est ici regulier, `n` rectangles de largeur `(b - a)/n`,
la hauteur prise a gauche : `xi_i = a + i (b - a)/n`. Pour une fonction
continue la limite ne depend pas de ce choix, et vaut `F(b) - F(a)`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import ENTIER, VARIABLE

INDICE: Final = sp.Symbol("i", integer=True, nonnegative=True)

type Bornes = tuple[object, object]


def somme_de_riemann(
    integrande: sp.Expr, bornes: Bornes, rectangles: object
) -> sp.Expr:
    """`sum f(a + i h) h` pour `i` de 0 a `n - 1`, `n` entier ou symbolique."""
    nombre = sp.sympify(rectangles)
    if nombre.is_number and nombre < 1:
        msg = f"au moins un rectangle est necessaire, {nombre} demande"
        raise ValueError(msg)
    bas, haut = (sp.sympify(b) for b in bornes)
    largeur = (haut - bas) / nombre
    hauteur = integrande.subs(VARIABLE, bas + INDICE * largeur)
    return sp.simplify(sp.summation(hauteur * largeur, (INDICE, 0, nombre - 1)))


def limite_des_sommes(integrande: sp.Expr, bornes: Bornes) -> sp.Expr:
    """La limite des sommes de Riemann quand le nombre de rectangles croit."""
    return sp.limit(somme_de_riemann(integrande, bornes, ENTIER), ENTIER, sp.oo)
