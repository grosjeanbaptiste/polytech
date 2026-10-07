"""Exercice cote 2025 (Mons), question 3 : deux branches en parallele.

La source alimente `R0` et `C`, parcourus par `i1` ; le courant se partage
ensuite entre la branche `R1`-`L` et la branche `R2`, que parcourt `i2`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

R0, R1, R2, L, C, W = sp.symbols("R_0 R_1 R_2 L C omega", positive=True)
E0: Final = sp.Symbol("E_0")
COURANTS: Final = sp.symbols("I_1 I_2")
RIEN: Final = sp.Integer(0)
"""Aucune impedance ajoutee a `R2`."""


def mailles(en_serie: sp.Expr = RIEN) -> tuple[sp.Expr, sp.Expr]:
    """Les deux mailles, chacune egalee a zero ; `en_serie` s'ajoute a `R2`."""
    i1, i2 = COURANTS
    derivation = (R1 + sp.I * W * L) * (i1 - i2)
    return (
        R0 * i1 + i1 / (sp.I * W * C) + derivation - E0,
        (R2 + en_serie) * i2 - derivation,
    )


def systeme() -> tuple[sp.Matrix, sp.Matrix]:
    """La matrice des impedances et le second membre."""
    matrice, constantes = sp.linear_eq_to_matrix(mailles(), COURANTS)
    return matrice, constantes


def rapport(en_serie: sp.Expr = RIEN) -> sp.Expr:
    """`I1/I2`, tire de la seule seconde maille."""
    i1, i2 = COURANTS
    (solution,) = sp.solve(mailles(en_serie)[1], i1)
    return sp.simplify(solution / i2)


def dephasage() -> sp.Expr:
    """L'argument de `I1/I2` : celui du numerateur moins celui du denominateur."""
    return sp.atan(W * L / (R1 + R2)) - sp.atan(W * L / R1)


def impedance_a_ajouter() -> sp.Expr:
    """`Z`, en serie avec `R2`, pour que `i2 = i1/2`."""
    inconnue = sp.Symbol("Z")
    (solution,) = sp.solve(sp.Eq(rapport(inconnue), 2), inconnue)
    return sp.expand(solution)


def est_realisable(*, r1: float, r2: float) -> bool:
    """Un dipole passif a une partie reelle positive ou nulle."""
    partie_reelle = sp.re(impedance_a_ajouter().subs({R1: r1, R2: r2}))
    return bool(partie_reelle >= 0)
