"""Exercice cote 2023 (Mons), question 3 : trois mailles couplees deux a deux.

La maille 1 (`E`, `R1`, `L1`) est couplee a la maille 2 (`R2`, `L2`, `L3`)
par `L12` ; la maille 2 l'est a la maille 3 (`R3`, `L4`, `L5`) par `L34`.
"""

from __future__ import annotations

import cmath
import math
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.bornes_pointees import (
    signe_de_la_mutuelle,
)

R1, R2, R3, E = sp.symbols("R_1 R_2 R_3 E", positive=True)
L1, L2, L3, L4, L5 = sp.symbols("L_1 L_2 L_3 L_4 L_5", positive=True)
L12, L34, P = sp.symbols("L_12 L_34 p", positive=True)
COURANTS: Final = sp.symbols("I_1 I_2 I_3")
RESISTANCE: Final = 1.0
"""`R1`, en ohms, a la troisieme sous-question."""
REACTANCE: Final = 2 * math.pi * 50 * 10e-3
"""`w L1`, en ohms, a 50 Hz."""
MUTUELLE: Final = 2 * math.pi * 50 * 1e-3
"""`w L12`, en ohms."""
SOURCE: Final = 5.0
"""La valeur efficace de `e`, de phase nulle."""
PRIMAIRE: Final = 2 + 3j
"""`I1`, en amperes efficaces, donne par l'enonce."""


def signes() -> tuple[int, int]:
    """`i1` et `i2` entrent par les points ; `i2` sort du point de `L3`, `i3` entre."""
    premiere = signe_de_la_mutuelle(premier_par_le_point=True, second_par_le_point=True)
    seconde = signe_de_la_mutuelle(premier_par_le_point=False, second_par_le_point=True)
    return premiere, seconde


def systeme() -> tuple[sp.Matrix, sp.Matrix]:
    """La matrice de Laplace et son second membre, conditions initiales nulles."""
    i1, i2, i3 = COURANTS
    premiere, seconde = signes()
    mailles = (
        (R1 + P * L1) * i1 + premiere * P * L12 * i2 - E / P,
        (R2 + P * (L2 + L3)) * i2 + premiere * P * L12 * i1 + seconde * P * L34 * i3,
        (R3 + P * (L4 + L5)) * i3 + seconde * P * L34 * i2,
    )
    matrice, constantes = sp.linear_eq_to_matrix(mailles, COURANTS)
    return matrice, constantes


def regime() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le theoreme de la valeur finale : `lim p I(p)` quand `p` tend vers zero."""
    matrice, constantes = systeme()
    solution = matrice.LUsolve(constantes)
    limites = (sp.limit(sp.simplify(P * terme), P, 0) for terme in solution)
    premier, deuxieme, troisieme = limites
    return premier, deuxieme, troisieme


def courant_du_secondaire() -> complex:
    """`I2`, tire de la maille du primaire : `E = (R1 + j w L1) I1 + j w L12 I2`."""
    chute = (RESISTANCE + 1j * REACTANCE) * PRIMAIRE
    return (SOURCE - chute) / (signes()[0] * 1j * MUTUELLE)


def sinusoide() -> tuple[float, float]:
    """L'amplitude de `i2(t)`, en amperes, et sa phase a l'origine, en radians."""
    courant = courant_du_secondaire()
    return abs(courant) * math.sqrt(2), cmath.phase(courant)


def residu_du_primaire() -> float:
    """`|E - (R1 + j w L1) I1 - j w L12 I2|`, qui doit etre nul."""
    induite = signes()[0] * 1j * MUTUELLE * courant_du_secondaire()
    return abs(SOURCE - (RESISTANCE + 1j * REACTANCE) * PRIMAIRE - induite)


def puissances() -> tuple[float, float, float]:
    """En watts : fournie par la source, dissipee dans `R1`, transmise au secondaire."""
    fournie = (SOURCE * PRIMAIRE.conjugate()).real
    dissipee = RESISTANCE * abs(PRIMAIRE) ** 2
    induite = signes()[0] * 1j * MUTUELLE * courant_du_secondaire()
    return fournie, dissipee, (induite * PRIMAIRE.conjugate()).real


def impedance_d_entree() -> complex:
    """`E/I1`, ce que la source voit."""
    return SOURCE / PRIMAIRE
