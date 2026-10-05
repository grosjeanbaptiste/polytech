"""Les suites de fonctions : convergence simple, convergence uniforme.

`f_n` converge **simplement** vers `f` si, en chaque `x`, la suite de
nombres `f_n(x)` tend vers `f(x)` : le rang `N(eps, x)` a partir duquel
`|f_n(x) - f(x)| < eps` peut dependre de `x`. Elle converge
**uniformement** si un meme rang convient pour tous les `x`, c'est-a-dire
si `d(f_n, f) = sup |f_n - f|` tend vers 0.

L'uniformite est ce qui autorise a passer a la limite : la limite de
fonctions continues est continue, et l'integrale sur un intervalle borne
passe a la limite. Sans elle, ni l'un ni l'autre ; et la derivee demande
la convergence uniforme des derivees elles-memes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.series.uniforme import distance_uniforme
from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence


def limite_simple(terme: sp.Expr, indice: sp.Symbol, point: object) -> sp.Expr:
    """`lim f_n(x)` en un point donne.

    Si `f_n(x)` change de forme avec `n`, c'est la forme valable pour les
    grands rangs qui donne la limite.
    """
    valeur = sp.sympify(terme).subs(VARIABLE, sp.sympify(point))
    if not isinstance(valeur, sp.Piecewise):
        return sp.limit(valeur, indice, sp.oo)
    reel = sp.Symbol("m", positive=True)
    morceaux = valeur.subs(indice, reel).as_expr_set_pairs(sp.Interval(1, sp.oo))
    (ultime,) = [expression for expression, rangs in morceaux if rangs.sup == sp.oo]
    return sp.limit(ultime, reel, sp.oo)


def distances_uniformes(
    terme: sp.Expr,
    limite: object,
    indice: sp.Symbol,
    intervalle: sp.Set,
    rangs: Sequence[int],
) -> tuple[sp.Expr, ...]:
    """`d(f_n, f) = sup |f_n - f|` pour chacun des rangs demandes."""
    return tuple(
        distance_uniforme(
            sp.sympify(terme).subs(indice, rang), limite, VARIABLE, intervalle
        )
        for rang in rangs
    )


def rang_de_la_puissance(epsilon: object, point: object) -> sp.Expr:
    """`N(eps, x) = ln(eps) / ln(x)` : `x^n < eps` des que `n` le depasse."""
    seuil, x = sp.sympify(epsilon), sp.sympify(point)
    hors_bornes = [v for v in (seuil, x) if v.is_number and not (v > 0 and v < 1)]
    if hors_bornes:
        msg = f"valeur attendue strictement entre 0 et 1, {hors_bornes[0]} recue"
        raise ValueError(msg)
    return sp.log(seuil) / sp.log(x)


def tente(rang: int) -> sp.Expr:
    """La tente de hauteur `n` en `1/n`, nulle hors de `[0, 2/n]` : aire 1."""
    n = sp.Integer(rang)
    return sp.Piecewise(
        (n**2 * VARIABLE, (VARIABLE >= 0) & (1 / n >= VARIABLE)),
        (n * (2 - n * VARIABLE), (1 / n < VARIABLE) & (2 / n >= VARIABLE)),
        (0, True),
    )
