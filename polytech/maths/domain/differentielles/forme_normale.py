"""Mettre une equation lineaire d'ordre `n` sous forme normale.

`a_n(t) y^(n) + ... + a_1(t) y' + a_0(t) y = x(t)`, avec `a_n` jamais nul,
devient un systeme du premier ordre par `y_1 = y, y_2 = y', ..., y_n =
y^(n-1)` : `y' = P(t) y + x(t)`, ou `P` porte des 1 au-dessus de sa
diagonale et `-a_k / a_n` sur sa derniere ligne, et ou seul le dernier
terme du second membre, `x / a_n`, n'est pas nul.

Le wronskien de ce systeme est fait de `n` solutions `g_i` et de leurs
derivees. La variation des constantes donne alors une solution
particuliere de l'equation :

    somme g_i(t) int_{t_0}^{t} W_ni(tau) x(tau) / (det W(tau) a_n(tau)) dtau,

`W_ni` etant le cofacteur de la derniere ligne.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.differentielles.wronskien import Wronskien
from polytech.maths.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Sequence


def matrice_compagnon(coefficients: Sequence[object]) -> sp.Matrix:
    """`P(t)`, pour les coefficients `a_0, a_1, ..., a_n`."""
    *autres, dominant = (sp.sympify(a) for a in coefficients)
    if dominant == 0:
        msg = "coefficient dominant nul : l'equation n'est pas d'ordre n"
        raise ValueError(msg)
    ordre = len(autres)
    matrice = sp.zeros(ordre, ordre)
    for ligne in range(ordre - 1):
        matrice[ligne, ligne + 1] = 1
    for colonne, coefficient in enumerate(autres):
        matrice[ordre - 1, colonne] = sp.simplify(-coefficient / dominant)
    return matrice


def force_normale(force: object, dominant: object, ordre: int) -> sp.Matrix:
    """`(0, ..., 0, x / a_n)`."""
    dernier = sp.simplify(sp.sympify(force) / sp.sympify(dominant))
    return sp.Matrix([*(0,) * (ordre - 1), dernier])


def particuliere_par_le_wronskien(
    solutions: Sequence[sp.Expr], *, dominant: object, force: object, t0: object
) -> sp.Expr:
    """La solution particuliere de la formule (6.15)."""
    muette = sp.Symbol("tau", positive=True)
    wronskien = Wronskien.des_solutions(solutions)
    derniere = len(solutions) - 1
    source = sp.sympify(force) / (sp.sympify(dominant) * wronskien.determinant())
    termes = []
    for i, solution in enumerate(solutions):
        integrande = (wronskien.matrice.cofactor(derniere, i) * source).subs(
            TEMPS, muette
        )
        termes.append(
            solution * sp.integrate(sp.simplify(integrande), (muette, t0, TEMPS))
        )
    return sp.simplify(sum(termes, sp.Integer(0)))
