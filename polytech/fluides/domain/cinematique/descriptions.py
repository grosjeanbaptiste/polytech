"""Les descriptions lagrangienne et eulerienne d'un meme mouvement.

Chapitre III.1 des planches. La description **lagrangienne** suit chaque
particule : `x(xi, t)` donne la position, a l'instant `t`, de la particule
qui portait l'etiquette `xi` au depart. Sa vitesse et son acceleration
sont `v_L = dx/dt` et `a_L = d2x/dt2`, a `xi` fixe.

La description **eulerienne** se place en un point : `xi(x, t)` nomme la
particule qui s'y trouve a l'instant `t`. Les deux vitesses sont la meme
grandeur, `v = v_L(xi(x, t), t) = v_E(x(xi, t), t)` : c'est celle du point
materiel `xi` qui occupe la position `x`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X


def vitesse_lagrangienne(position: sp.Expr) -> sp.Expr:
    """`v_L(xi, t) = dx/dt`, l'etiquette etant tenue fixe."""
    return sp.diff(position, T)


def acceleration_lagrangienne(position: sp.Expr) -> sp.Expr:
    """`a_L(xi, t) = d2x/dt2`, l'etiquette etant tenue fixe."""
    return sp.diff(position, T, 2)


def etiquette(position: sp.Expr, nom: sp.Symbol) -> sp.Expr:
    """`xi(x, t)` : la particule qui occupe l'abscisse `x` a l'instant `t`."""
    solutions = sp.solve(sp.Eq(position, X), nom)
    if len(solutions) != 1:
        msg = f"pas une etiquette unique : {len(solutions)} particules en x"
        raise ValueError(msg)
    return sp.simplify(solutions[0])


def vitesse_eulerienne(position: sp.Expr, nom: sp.Symbol) -> sp.Expr:
    """`v_E(x, t) = v_L(xi(x, t), t)`."""
    lagrangienne = vitesse_lagrangienne(position)
    return sp.simplify(lagrangienne.subs(nom, etiquette(position, nom)))
