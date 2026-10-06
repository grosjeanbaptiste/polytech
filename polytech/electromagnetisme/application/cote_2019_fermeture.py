"""Exercice cote 2019, question 5 : fermer une branche R-C sur un circuit inductif.

Avant `t = 0`, le courant `E/(R1 + R2)` traverse `L1`, `R1`, `L2`, `R2`.
La fermeture branche `R3`-`C3` en parallele sur `L2`-`R2`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    valeur_finale,
    valeur_initiale,
)

P: Final = VARIABLE
E, R1, R2, R3 = sp.symbols("E R_1 R_2 R_3", positive=True)
L1, L2, C = sp.symbols("L_1 L_2 C_3", positive=True)
IP, I1P, UC = sp.symbols("I I_1 U_C")
"""Les images `I(p)`, `I1(p)` et `U_C(p)`."""


def conditions_initiales() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`i(0)`, `i1(0)` et `u_C(0)` : interrupteur ouvert, regime etabli."""
    return sp.Integer(0), E / (R1 + R2), sp.Integer(0)


def systeme() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Les equations (1), (2) et (3) du manuscrit, chacune egalee a zero."""
    i0, i10, u0 = conditions_initiales()
    d_i, d_i1 = P * IP - i0, P * I1P - i10
    grande_maille = L1 * (d_i + d_i1) + R1 * (IP + I1P) + L2 * d_i1 + R2 * I1P - E / P
    petite_maille = R3 * IP - R2 * I1P - L2 * d_i1 + UC
    condensateur = -IP + C * (P * UC - u0)
    return sp.expand(grande_maille), sp.expand(petite_maille), sp.expand(condensateur)


def images() -> dict[sp.Symbol, sp.Expr]:
    """`I(p)`, `I1(p)` et `U_C(p)`, solutions du systeme."""
    solution: dict[sp.Symbol, sp.Expr] = sp.solve(systeme(), (IP, I1P, UC))
    return solution


def regime() -> tuple[sp.Expr, sp.Expr]:
    """Le courant `i` et la tension de `C3` quand `t` tend vers l'infini."""
    resolues = images()
    return sp.simplify(valeur_finale(resolues[IP])), sp.simplify(
        valeur_finale(resolues[UC])
    )


def depart_du_courant() -> tuple[sp.Expr, sp.Expr]:
    """`i(0+)` et sa pente, par le theoreme de la valeur initiale."""
    image = images()[IP]
    depart = sp.simplify(valeur_initiale(image))
    return depart, sp.simplify(valeur_initiale(P * image - depart))
