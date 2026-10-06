"""Exercice cote 2023 (Charleroi), question 3 : une bobine sur deux condensateurs.

Avant `t = 0` la bobine est en serie avec la source et `R1`, en regime
sinusoidal ; apres, elle se referme sur `C1`, avec `R2`-`C2` en parallele.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp
from scipy.integrate import solve_ivp

from polytech.electromagnetisme.domain.circuits.phaseurs import valeur_instantanee
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    valeur_finale,
)

if TYPE_CHECKING:
    import numpy as np

T: Final = TEMPS
P: Final = VARIABLE
E, R1, R2, L, W = sp.symbols("E R_1 R_2 L omega", positive=True)
C1, C2 = sp.symbols("C_1 C_2", positive=True)
THETA = sp.Symbol("theta", real=True)
I10: Final = sp.Symbol("i_10", real=True)
"""`i1(0)`, laisse en lettre dans le systeme."""
INCONNUES: Final = sp.symbols("U_c1 U_c2 I_1 I_2")


def equations() -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Les quatre equations transformees, condensateurs initialement decharges."""
    u1, u2, i1, i2 = INCONNUES
    return (
        u1 + L * (P * i1 - I10),
        u1 - u2 - R2 * i2,
        C1 * P * u1 - i1 + i2,
        C2 * P * u2 - i2,
    )


def systeme() -> tuple[sp.Matrix, sp.Matrix]:
    """La matrice et le second membre, dans l'ordre du manuscrit."""
    matrice, constantes = sp.linear_eq_to_matrix(equations(), INCONNUES)
    return matrice, constantes


def courant_avant() -> sp.Expr:
    """`i1(t)` avant le basculement : `E = -(R1 + j omega L) I1`, au sens de `i1`."""
    phaseur = -E * sp.exp(sp.I * THETA) / (R1 + sp.I * W * L)
    return valeur_instantanee(phaseur, W)


def courant_initial() -> sp.Expr:
    """`i1(0)`, que la bobine conserve au basculement."""
    return courant_avant().subs(T, 0)


def regime() -> tuple[sp.Expr, ...]:
    """Les quatre grandeurs quand `t` tend vers l'infini."""
    matrice, constantes = systeme()
    images = matrice.LUsolve(constantes)
    return tuple(sp.simplify(valeur_finale(image)) for image in images)


def bilan_d_energie_chiffre() -> tuple[float, float]:
    """Sur un exemple : l'energie dissipee dans `R2`, et celle de la bobine au depart.

    Integration numerique des equations d'etat, avec toutes les valeurs
    a l'unite : un chemin qui ne doit rien a la transformee de Laplace.
    """

    def derivees(_: float, etat: np.ndarray) -> list[float]:
        courant, premiere, seconde, _ = etat
        dans_r2 = premiere - seconde
        return [-premiere, courant - dans_r2, dans_r2, dans_r2**2]

    solution = solve_ivp(
        derivees, (0.0, 400.0), [1.0, 0.0, 0.0, 0.0], rtol=1e-11, atol=1e-13
    )
    return float(solution.y[3, -1]), 0.5
