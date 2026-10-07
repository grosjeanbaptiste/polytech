"""Exercice cote 2024 (Mons), question 3 : deux interrupteurs basculent ensemble.

Avant : `E1` alimente les branches `R11`-`C` et `R12`-`L1`, `E2` le
secondaire `L2`-`R2`. Apres : le primaire est la maille `C`, `R11`,
`R12`, `L1`, et le secondaire se referme sur un fil.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.reseau import (
    Condensateur,
    Inductance,
    Reseau,
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.magnetostatique.bornes_pointees import (
    signe_de_la_mutuelle,
)

R11, R12, R2, E1, E2 = sp.symbols("R_11 R_12 R_2 E_1 E_2", positive=True)
L1, L2, L12, C, P = sp.symbols("L_1 L_2 L_12 C p", positive=True)
INCONNUES: Final = sp.symbols("I_1 I_2 U_C")
EXEMPLE: Final = {R11: 2, R12: 3, R2: 5, E1: 7, E2: 11, L1: 4, L2: 9, L12: 1, C: 6}
"""Des valeurs quelconques, pour les limites ; `L12^2 < L1 L2`."""


def avant() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`(i1, i2, uC)` juste avant : `i1` monte dans `L1`, le courant y descend."""
    return -E1 / R12, E2 / R2, E1


def avant_par_le_reseau() -> tuple[sp.Expr, sp.Expr]:
    """`(i1, uC)` du primaire, resolu comme un reseau en regime."""
    circuit = Reseau(
        (
            SourceDeTension("haut", "bas", E1),
            Resistance("haut", "sur_c", R11),
            Condensateur("sur_c", "bas"),
            Resistance("haut", "sur_l", R12),
            Inductance("sur_l", "bas"),
        ),
        "bas",
    )
    potentiels = circuit.potentiels()
    return (potentiels["sur_l"] - potentiels["haut"]) / R12, potentiels["sur_c"]


def signe() -> int:
    """`i1` entre dans `L1` par le point ; `i2` sort de `L2` par le sien."""
    return signe_de_la_mutuelle(premier_par_le_point=True, second_par_le_point=False)


Triplet = tuple[sp.Expr, sp.Expr, sp.Expr]


def _mailles(etat: Triplet, pentes: Triplet) -> Triplet:
    """Primaire, secondaire, condensateur : chacune egalee a zero."""
    i1, i2, uc = etat
    d1, d2, du = pentes
    mutuelle = signe() * L12
    return (
        (R11 + R12) * i1 + L1 * d1 + mutuelle * d2 + uc,
        R2 * i2 + L2 * d2 + mutuelle * d1,
        i1 - C * du,
    )


def systeme() -> tuple[sp.Matrix, sp.Matrix]:
    """Les trois equations de Laplace, conditions initiales a droite."""
    i1, i2, uc = INCONNUES
    depart_un, depart_deux, depart_c = avant()
    pentes = (P * i1 - depart_un, P * i2 - depart_deux, P * uc - depart_c)
    matrice, constantes = sp.linear_eq_to_matrix(_mailles(INCONNUES, pentes), INCONNUES)
    return matrice, constantes


def limites(point: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`lim p X(p)` en `point` : zero donne le regime, l'infini l'etat initial."""
    matrice, constantes = systeme()
    solution = matrice.subs(EXEMPLE).LUsolve(constantes.subs(EXEMPLE))
    premier, deuxieme, troisieme = (
        sp.limit(sp.simplify(P * terme), P, point) for terme in solution
    )
    return premier, deuxieme, troisieme


def residu_du_bilan_d_energie() -> sp.Expr:
    """`dW/dt + (R11 + R12) i1^2 + R2 i2^2`, nul si les equations sont coherentes."""
    i1, i2, uc, d1, d2, du = sp.symbols("i_1 i_2 u_C d_1 d_2 d_u")
    pentes = sp.solve(_mailles((i1, i2, uc), (d1, d2, du)), (d1, d2, du))
    couplee = signe() * L12 * (d1 * i2 + i1 * d2)
    stockee = L1 * i1 * d1 + L2 * i2 * d2 + couplee + C * uc * du
    dissipee = (R11 + R12) * i1**2 + R2 * i2**2
    return sp.simplify(stockee.subs(pentes) + dissipee)
