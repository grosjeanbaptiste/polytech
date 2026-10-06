"""Exercice cote 2018, question 2 : deux fils, et ou leur champ est vertical.

Le champ d'un fil est tangent aux cercles qui l'entourent : il s'obtient
en tournant d'un quart de tour le vecteur qui va du fil au point. Ce
produit vectoriel remplace les angles du manuscrit.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.superposition import Fil

I1, I2 = sp.symbols("I_1 I_2", positive=True)
X1, X2 = sp.symbols("x_1 x_2", positive=True)
Y = sp.Symbol("y", real=True)


def _champ_d_un_fil(courant: sp.Expr, abscisse: sp.Expr) -> sp.Matrix:
    """Le champ en `(0, y)` d'un fil en `(abscisse, 0)`, courant compte sortant."""
    rayon = sp.Matrix([-abscisse, Y])
    distance = sp.sqrt(rayon.dot(rayon))
    tangent = sp.Matrix([-rayon[1], rayon[0]]) / distance
    return Fil(courant, distance).module_du_champ() * tangent


def champ() -> tuple[sp.Expr, sp.Expr]:
    """`(H_x, H_y)` sur l'axe `y` : `I1` sortant en `-x1`, `I2` rentrant en `x2`."""
    total = _champ_d_un_fil(I1, -X1) + _champ_d_un_fil(-I2, X2)
    return sp.simplify(total[0]), sp.simplify(total[1])


def points_d_alignement() -> tuple[sp.Expr, sp.Expr]:
    """Les zeros de `H_x` : `y = 0`, puis le carre des deux autres ordonnees."""
    carre = sp.Symbol("u")
    numerateur = sp.numer(sp.together(champ()[0] / Y)).subs(Y**2, carre)
    (solution,) = sp.solve(sp.expand(numerateur).subs(Y**2, carre), carre)
    return sp.Integer(0), sp.simplify(solution)


def seuil_d_existence() -> sp.Expr:
    """La valeur de `x2 / x1` au-dela de laquelle les deux autres points existent."""
    rapport = sp.Symbol("r", positive=True)
    carre = points_d_alignement()[1].subs(X2, rapport * X1)
    (seuil,) = sp.solve(sp.numer(sp.together(carre)), rapport)
    return seuil
