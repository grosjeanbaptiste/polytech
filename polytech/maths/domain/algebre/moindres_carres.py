"""Resoudre un systeme impossible au sens des moindres carres ; la regression.

Le syllabus d'algebre (IX.6) : `A X = b` a une solution si et seulement si
`b` est dans le sous-espace `W` engendre par les colonnes de `A`. S'il n'y
est pas, « on cherche un `X` tel que `||b - A X||` soit la plus petite
possible », ce qui revient a resoudre `A X = proj_W b`.

`b - A X` est alors orthogonal a chaque colonne de `A` : `A^t (b - A X) = 0`,
soit les **equations normales** `A^t A X = A^t b`. Elles ont une solution
unique quand les colonnes de `A` sont libres.

La **regression lineaire** en est l'application : la droite `y = a x + b`
la plus proche de points `(x_i, y_i)` rend minimale la somme des carres
des ecarts verticaux `(a x_i + b) - y_i`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


def au_sens_des_moindres_carres(
    matrice: sp.Matrix, second_membre: sp.Matrix
) -> tuple[sp.Expr, ...]:
    """Le `X` qui rend `||b - A X||` minimale, par les equations normales."""
    normale = matrice.T * matrice
    if normale.det() == 0:
        msg = (
            "colonnes liees : la solution au sens des moindres carres n'est pas unique"
        )
        raise ValueError(msg)
    return tuple(normale.inv() * matrice.T * second_membre)


def droite_de_regression(
    points: Sequence[tuple[object, object]],
) -> tuple[sp.Expr, sp.Expr]:
    """`(a, b)` tels que `y = a x + b` passe au plus pres des points."""
    matrice = sp.Matrix([[x, 1] for x, _ in points])
    ordonnees = sp.Matrix([y for _, y in points])
    pente, ordonnee = au_sens_des_moindres_carres(matrice, ordonnees)
    return pente, ordonnee


def projection_et_residu(
    matrice: sp.Matrix, second_membre: sp.Matrix
) -> tuple[sp.Matrix, sp.Matrix]:
    """`proj_W b` et `b - proj_W b`, `W` engendre par les colonnes, liees ou non.

    Une base orthogonale de `W` (Gram-Schmidt sur les colonnes libres) donne
    la projection : elle existe et est unique meme quand les colonnes sont
    liees, la ou les equations normales n'ont plus de solution unique.
    """
    projete = sp.zeros(*second_membre.shape)
    for colonne in sp.GramSchmidt(matrice.columnspace()):
        projete += (colonne.dot(second_membre) / colonne.dot(colonne)) * colonne
    projete = sp.simplify(projete)
    return projete, sp.simplify(second_membre - projete)
