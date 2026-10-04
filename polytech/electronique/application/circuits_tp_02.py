"""Les circuits du TP 2 d'electronique, par les formules du cours.

Serie, parallele, diviseur de tension et superposition : rien d'autre.
Les schemas sont lus sur l'enonce ; deux points s'y lisent mal et sont
fixes ici. A l'exercice 3.1, le court-circuit est en parallele de `R`.
A l'exercice 5.3, la fleche de `i_s` pointe vers la masse : la source
**tire** le courant du noeud de sortie, d'ou le signe moins.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application.symboles_tp_02 import (
    CL,
    I1,
    I2,
    IS,
    ITOT,
    OMEGA,
    R1,
    R2,
    R3,
    RL,
    RS,
    V1,
    V2,
    C,
    E,
    L,
    R,
    S,
    Z,
)
from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedances_en_parallele,
    impedances_en_serie,
)


def impedances_du_2() -> dict[str, sp.Expr]:
    """Les quatre 1-ports des exercices 2.1 a 2.3."""
    zl, zc = impedance_de_bobine(L, S), impedance_de_condensateur(C, S)
    return {
        "2.1a": impedances_en_parallele(zc, R),
        "2.1b": impedances_en_serie(R, zl),
        "2.2": sp.factor(impedances_en_serie(zl, impedances_en_parallele(R, zc))),
        "2.3": sp.factor(impedances_en_parallele(impedances_en_serie(R, zl), zc)),
    }


def courants_du_3_1() -> tuple[sp.Expr, sp.Expr]:
    """Un court-circuit en parallele de `R` emporte tout : `I_R = 0`."""
    return sp.Integer(0) * ITOT, ITOT


def transferts_du_4_1() -> dict[str, sp.Expr]:
    """Les trois diviseurs : en continu (en volts), le R-C, la charge."""
    continu = sp.Rational(12, 10) * diviseur_de_tension(
        sp.Integer(1000), sp.Integer(5000)
    )
    return {
        "continu": continu,
        "rc": diviseur_de_tension(impedance_de_condensateur(C, S), R),
        "charge": diviseur_de_tension(RS, impedances_en_parallele(Z, RL)),
    }


def sortie_du_5_1() -> sp.Expr:
    """La somme des deux contributions, l'autre source eteinte chaque fois."""
    une = V1 * diviseur_de_tension(R1, impedances_en_parallele(R1, R2))
    autre = V2 * diviseur_de_tension(R1, impedances_en_parallele(R1, R2))
    return sp.simplify(une + autre)


def noeuds_du_5_2() -> dict[int, sp.Expr]:
    """Tout le courant finit dans `C` : il traverse `R2`, et `R1` porte `I1`."""
    trois = (I1 + I2) * impedance_de_condensateur(C, S)
    deux = trois + R2 * (I1 + I2)
    return {1: sp.simplify(deux + R1 * I1), 2: sp.simplify(deux), 3: sp.simplify(trois)}


def contributions_du_5_3() -> dict[str, sp.Expr]:
    """Le continu de 3 V, le phaseur sous `E`, le phaseur sous `I_s`."""
    jw = sp.I * OMEGA
    branche = impedances_en_serie(R1, 1 / (jw * C))
    charge = impedances_en_parallele(R2, R3 + jw * L)
    return {
        "continu": 3 * diviseur_de_tension(R2, R3),
        "tension": sp.simplify(E * diviseur_de_tension(branche, charge)),
        "courant": sp.simplify(-IS * impedances_en_parallele(branche, charge)),
    }


def transferts_du_6_1() -> dict[str, sp.Expr]:
    """Les trois `H(j omega)` de l'exercice 6.1."""
    jw = sp.I * OMEGA
    return {
        "a": diviseur_de_tension(1 / (jw * C), R),
        "b": diviseur_de_tension(R, impedances_en_parallele(1 / (jw * (C + CL)), RL)),
        "c": diviseur_de_tension(R + RS, impedances_en_parallele(1 / (jw * C), RL)),
    }
