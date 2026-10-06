"""Le residu d'une fonction en une singularite isolee, et le theoreme des residus.

Le residu de `f` en `a` est l'unique `R` tel que `f(z) - R / (z - a)` soit
la derivee d'une fonction holomorphe autour de `a` ; c'est le coefficient
de `1 / (z - a)` de sa serie de Laurent. Il se calcule selon la nature de
la singularite :

1. reguliere : le residu est nul ;
2. pole d'ordre 1 : `lim f(z) (z - a)` ;
3. pole d'ordre `h` : `g^(h-1)(a) / (h - 1)!`, ou `g = f (z - a)^h` ;
4. quotient `g / h` : voir `quotient.py` ;
5. essentielle : la serie de Laurent, voir `laurent.py`.

Theoreme des residus : l'integrale de `f` sur une courbe simple fermee,
orientee positivement, vaut `2 pi i` fois la somme des residus aux
singularites qu'elle entoure.

Source : Mathematiques pour l'ingenieur III, syllabus, sections 9.1 et 9.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.analyse_complexe.laurent import residu_par_laurent
from polytech.maths.domain.analyse_complexe.singularites import ordre_du_pole

if TYPE_CHECKING:
    from collections.abc import Iterable


def limite_du_produit(fonction: sp.Expr, variable: sp.Symbol, point: object) -> sp.Expr:
    """`lim f(z) (z - a)` quand `z` tend vers `a`, sans juger de sa valeur."""
    a = sp.sympify(point)
    return sp.simplify(sp.limit((variable - a) * fonction, variable, a))


def residu_d_un_pole_simple(
    fonction: sp.Expr, variable: sp.Symbol, point: object
) -> sp.Expr:
    """Methode 2 : la limite du produit ; nulle si la singularite est reguliere."""
    limite = limite_du_produit(fonction, variable, point)
    if limite.is_infinite or limite.has(sp.oo, sp.zoo, sp.nan):
        msg = f"{point} n'est pas un pole simple de {fonction} : le produit diverge"
        raise ValueError(msg)
    return limite


def residu_d_un_pole(
    fonction: sp.Expr, variable: sp.Symbol, point: object, *, ordre: int
) -> sp.Expr:
    """Methode 3 : `g^(h-1)(a) / (h - 1)!`, avec `g = f (z - a)^h`."""
    if ordre < 1:
        msg = f"ordre {ordre} : un pole est d'ordre 1 au moins"
        raise ValueError(msg)
    a = sp.sympify(point)
    g = sp.simplify(fonction * (variable - a) ** ordre)
    derivee = sp.diff(g, variable, ordre - 1)
    return sp.simplify(sp.limit(derivee, variable, a) / sp.factorial(ordre - 1))


def residu(fonction: sp.Expr, variable: sp.Symbol, point: object) -> sp.Expr:
    """Le residu, par la methode que la nature de la singularite appelle."""
    try:
        ordre = ordre_du_pole(fonction, variable, point)
    except ValueError:
        return residu_par_laurent(fonction, variable, point)
    if ordre == 0:
        return sp.Integer(0)
    return residu_d_un_pole(fonction, variable, point, ordre=ordre)


def integrale_par_les_residus(
    fonction: sp.Expr, variable: sp.Symbol, singularites: Iterable[object]
) -> sp.Expr:
    """`2 pi i` fois la somme des residus aux singularites entourees."""
    somme = sum((residu(fonction, variable, a) for a in singularites), sp.Integer(0))
    return sp.simplify(2 * sp.pi * sp.I * somme)
