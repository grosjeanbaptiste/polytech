"""Coordonnees dans un espace de fonctions de dimension finie, matrice d'un operateur.

Le syllabus d'algebre (VIII.7, exercices 18 et 22) fait agir la
derivation sur l'espace engendre par quelques fonctions -- `{e^t, t e^t}`,
`{sin t, cos t}` -- et demande sa matrice dans cette base : la `j`-ieme
colonne porte les coordonnees de l'image de la `j`-ieme fonction.

Une egalite `g = c_1 b_1 + ... + c_n b_n` entre fonctions se derive : en
un point, la fonction et ses `n - 1` premieres derivees donnent `n`
equations lineaires en les `c_i`. Si leur matrice est inversible, les
coefficients sont determines ; il reste a verifier que l'egalite vaut
partout, car `g` peut sortir de l'espace engendre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def coordonnees_d_une_fonction(
    fonction: sp.Expr,
    base: Sequence[sp.Expr],
    variable: sp.Symbol,
    point: object = 0,
) -> tuple[sp.Expr, ...]:
    """Les `c_i` tels que `fonction = somme(c_i base_i)`, identiquement."""
    ordre = len(base)

    def jet(expression: sp.Expr, rang: int) -> sp.Expr:
        return sp.diff(expression, variable, rang).subs(variable, point)

    jets = sp.Matrix(ordre, ordre, lambda k, i: jet(base[i], k))
    if sp.simplify(jets.det()) == 0:
        msg = "les derivees en ce point ne separent pas la base : changer de point"
        raise ValueError(msg)
    coefficients = jets.inv() * sp.Matrix([jet(fonction, k) for k in range(ordre)])
    combinaison = sum(c * b for c, b in zip(coefficients, base, strict=True))
    if sp.simplify(combinaison - fonction) != 0:
        msg = "fonction hors de l'espace engendre par la base"
        raise ValueError(msg)
    return tuple(sp.simplify(c) for c in coefficients)


def matrice_d_un_operateur(
    operateur: Callable[[sp.Expr], sp.Expr],
    base: Sequence[sp.Expr],
    variable: sp.Symbol,
    point: object = 0,
) -> sp.Matrix:
    """La matrice dont chaque colonne porte les coordonnees de l'image d'une base."""
    colonnes = [
        coordonnees_d_une_fonction(operateur(fonction), base, variable, point)
        for fonction in base
    ]
    return sp.Matrix(colonnes).T
