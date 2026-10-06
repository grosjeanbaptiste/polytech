"""Exercice cote 2016, question 3 : deux circuits couples, en Laplace.

Litteral, comme l'enonce. La branche `R11`-`C1` est en parallele sur la
source ideale `E1` : elle ne change rien a `i1`, et le manuscrit l'omet a
bon droit.
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
E1, E2 = sp.symbols("E_1 E_2", positive=True)
R12, R2 = sp.symbols("R_12 R_2", positive=True)
L1, L2, L12 = sp.symbols("L_1 L_2 L_12", positive=True)
I1, I2 = sp.symbols("I_1 I_2")
"""Les images `I1(p)` et `I2(p)`."""


def conditions_initiales() -> tuple[sp.Expr, sp.Expr]:
    """Avant la fermeture : `L1` est un fil, le secondaire est ouvert."""
    return E1 / R12, sp.Integer(0)


def _derivee(image: sp.Expr, initiale: sp.Expr) -> sp.Expr:
    """`L(di/dt) = p I(p) - i(0)`."""
    return P * image - initiale


def systeme_en_laplace() -> tuple[sp.Expr, sp.Expr]:
    """Les deux mailles transformees, chacune egalee a zero."""
    depart_1, depart_2 = conditions_initiales()
    d1, d2 = _derivee(I1, depart_1), _derivee(I2, depart_2)
    primaire = R12 * I1 + L1 * d1 + L12 * d2 - E1 / P
    secondaire = R2 * I2 + L2 * d2 + L12 * d1 - E2 / P
    return sp.expand(primaire), sp.expand(secondaire)


def images() -> tuple[sp.Expr, sp.Expr]:
    """`I1(p)` et `I2(p)`, solutions du systeme."""
    resolues = sp.solve(systeme_en_laplace(), (I1, I2))
    return sp.simplify(resolues[I1]), sp.simplify(resolues[I2])


def regime() -> tuple[sp.Expr, sp.Expr]:
    """Les deux courants quand `t` tend vers l'infini."""
    premiere, seconde = images()
    return sp.simplify(valeur_finale(premiere)), sp.simplify(valeur_finale(seconde))


def valeurs_initiales() -> tuple[sp.Expr, sp.Expr]:
    """Les deux courants juste apres la fermeture."""
    premiere, seconde = images()
    return sp.simplify(valeur_initiale(premiere)), sp.simplify(valeur_initiale(seconde))


def pente_initiale_du_secondaire() -> sp.Expr:
    """`di2/dt` en `0+` : la valeur initiale de `p I2(p) - i2(0)`."""
    seconde = images()[1]
    return sp.simplify(valeur_initiale(_derivee(seconde, conditions_initiales()[1])))
