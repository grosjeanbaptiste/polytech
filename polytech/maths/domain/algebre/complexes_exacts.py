"""Les nombres complexes en ecriture exacte : module, argument, Moivre, racines.

`z = |z| (cos theta + i sin theta) = |z| e^(i theta)`. Sous cette forme le
produit multiplie les modules et ajoute les arguments, d'ou la formule de
Moivre `z^n = |z|^n e^(i n theta)` et les `n` racines `n`-iemes

    y_l = |z|^(1/n) e^(i (theta + 2 l pi) / n),     l = 0, ..., n - 1,

sommets d'un polygone regulier. Une equation du second degre a
coefficients complexes a toujours deux racines, `(-b +- racine(Delta)) / 2a`,
ou `racine(Delta) = a + bi` se cherche par `a^2 - b^2 = Re Delta`,
`2ab = Im Delta`.

Ce module calcule en exact (radicaux, multiples de `pi`) ; `complexes.py`
porte les memes notions en flottants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Callable

ORDONNEE = sp.Symbol("y", real=True)
"""La partie imaginaire de `z = x + iy` dans le plan de Gauss."""


def forme_polaire(z: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(|z|, theta)` avec `theta` dans `]-pi, pi]` ; leve pour `z = 0`."""
    simplifie = sp.simplify(sp.expand_complex(sp.sympify(z)))
    if simplifie == 0:
        msg = "argument de zero : toutes les directions le representent"
        raise ValueError(msg)
    return sp.simplify(sp.Abs(simplifie)), sp.simplify(sp.arg(simplifie))


def par_moivre(z: sp.Expr, puissance: int) -> sp.Expr:
    """`z^n = |z|^n e^(i n theta)`."""
    module, angle = forme_polaire(z)
    return module**puissance * sp.exp(sp.I * puissance * angle)


def racines_de_l_unite(ordre: int) -> tuple[sp.Expr, ...]:
    """`eta_l = e^(2 i l pi / n)`, pour `l = 0, ..., n - 1`."""
    return tuple(sp.exp(2 * sp.I * sp.pi * sp.Rational(k, ordre)) for k in range(ordre))


def racines_niemes_exactes(z: sp.Expr, ordre: int) -> tuple[sp.Expr, ...]:
    """Les `n` racines `n`-iemes de `z`, la premiere multipliee par chaque `eta_l`."""
    module, angle = forme_polaire(z)
    premiere = sp.root(module, ordre) * sp.exp(sp.I * angle / ordre)
    return tuple(premiere * eta for eta in racines_de_l_unite(ordre))


def racines_carrees(delta: sp.Expr) -> tuple[sp.Expr, ...]:
    """Les deux `a + bi` de carre `Delta` : `a^2 - b^2 = Re Delta`, `2ab = Im Delta`."""
    a, b = sp.symbols("a b", real=True)
    valeur = sp.sympify(delta)
    systeme = [sp.Eq(a**2 - b**2, sp.re(valeur)), sp.Eq(2 * a * b, sp.im(valeur))]
    return tuple(s[a] + s[b] * sp.I for s in sp.solve(systeme, [a, b], dict=True))


def racines_du_second_degre(a: object, b: object, c: object) -> tuple[sp.Expr, sp.Expr]:
    """`(-b +- racine(Delta)) / 2a`, pour des coefficients complexes."""
    dominant, lineaire = sp.sympify(a), sp.sympify(b)
    if dominant == 0:
        msg = "coefficient dominant nul : ce n'est pas un second degre"
        raise ValueError(msg)
    racine = racines_carrees(sp.expand(lineaire**2 - 4 * dominant * sp.sympify(c)))[0]
    return (
        sp.simplify((-lineaire + racine) / (2 * dominant)),
        sp.simplify((-lineaire - racine) / (2 * dominant)),
    )


def lieu_dans_le_plan_de_gauss(relation: Callable[[sp.Expr], sp.Expr]) -> sp.Expr:
    """La condition ecrite en `x` et `y`, apres avoir pose `z = x + iy`."""
    return sp.expand(sp.simplify(relation(VARIABLE + sp.I * ORDONNEE)))
