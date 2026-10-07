"""Exercice cote 2022 (Mons), question 3 : trois mailles, deux bobines couplees.

La source alimente `R0` ; la branche `R1`-`C` porte `i0 - i1` ; la branche
`L1`-`R2` porte `i1` et induit dans le secondaire `L2`-`R3`, que parcourt
`i2`.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.bornes_pointees import (
    signe_de_la_mutuelle,
)

R0, R1, R2, R3 = sp.symbols("R_0 R_1 R_2 R_3", positive=True)
L1, L2, L12, C, W = sp.symbols("L_1 L_2 L_12 C omega", positive=True)
E0: Final = sp.Symbol("E_0")
COURANTS: Final = sp.symbols("I_0 I_1 I_2")
FREQUENCE: Final = 50.0
DONNEES: Final = {R2: 2, R3: 3, L1: 1e-3, L2: 2e-3, L12: 1e-4}
"""Les valeurs de la seconde sous-question, en ohms et en henrys."""


def signe() -> int:
    """`i1` entre dans `L1` par la borne sans point, `i2` dans `L2` par le point."""
    return signe_de_la_mutuelle(premier_par_le_point=False, second_par_le_point=True)


def mailles() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Les trois mailles, chacune egalee a zero."""
    i0, i1, i2 = COURANTS
    derivation = (R1 + 1 / (sp.I * W * C)) * (i0 - i1)
    mutuelle = signe() * sp.I * W * L12
    return (
        R0 * i0 + derivation - E0,
        -derivation + (sp.I * W * L1 + R2) * i1 + mutuelle * i2,
        (R3 + sp.I * W * L2) * i2 + mutuelle * i1,
    )


def systeme() -> tuple[sp.Matrix, sp.Matrix]:
    """La matrice des impedances et le second membre."""
    matrice, constantes = sp.linear_eq_to_matrix(mailles(), COURANTS)
    return matrice, constantes


def rapport() -> sp.Expr:
    """`I2/I1`, tire de la seule maille du secondaire."""
    _, i1, i2 = COURANTS
    (solution,) = sp.solve(mailles()[2], i2)
    return sp.simplify(solution / i1)


def rapport_chiffre() -> complex:
    """`I2/I1` a 50 Hz, avec les valeurs de l'enonce."""
    return complex(rapport().subs(DONNEES).subs(W, 2 * math.pi * FREQUENCE))


def dephasage_en_degres() -> float:
    """L'avance de `i2` sur `i1`, en degres."""
    valeur = rapport_chiffre()
    return math.degrees(math.atan2(valeur.imag, valeur.real))


def courants_resolus(
    *, r0: float, r1: float, capacite: float
) -> tuple[complex, complex]:
    """`I1` et `I2`, les trois mailles resolues.

    L'enonce ne donne ni `R0`, ni `R1`, ni `C` : on les fournit.
    """
    valeurs = {
        **DONNEES,
        R0: r0,
        R1: r1,
        C: capacite,
        E0: 1,
        W: 2 * math.pi * FREQUENCE,
    }
    matrice, constantes = systeme()
    solution = matrice.subs(valeurs).LUsolve(constantes.subs(valeurs))
    return complex(solution[1]), complex(solution[2])
