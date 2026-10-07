"""Exercice cote 2025 (Mons), question 1 : Thevenin, puis une pile et une resistance.

En regime `C` ouvre la branche `R1`, `E1`, `R3`, et `L` est un fil : il
reste, entre A et B, la source de courant, la branche `E2`-`R2` et `R4`.
Le pole positif de `E2` est du cote de `R2` ; celui de `E'`, du cote de A.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.equivalents import (
    resistance_vue_des_bornes,
    tension_a_vide,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Condensateur,
    Element,
    Inductance,
    Reseau,
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)

R2: Final = sp.Integer(2)
R4: Final = sp.Integer(4)
E2: Final = sp.Integer(10)
DEBIT: Final = sp.Integer(2)
RESISTANCE_DU_DIPOLE: Final = sp.Rational(5, 3)
PILE_DU_DIPOLE: Final = sp.Integer(5)


def _elements(e1: int, r3: int) -> tuple[Element, ...]:
    return (
        SourceDeCourant("B", "A", DEBIT),
        SourceDeTension("sur_r2", "A", E2),
        Resistance("sur_r2", "B", R2),
        Inductance("A", "sur_r4"),
        Resistance("sur_r4", "B", R4),
        Resistance("A", "apres_r1", sp.Integer(1)),
        SourceDeTension("apres_r1", "sur_c", sp.Integer(e1)),
        Condensateur("sur_c", "sur_r3"),
        Resistance("sur_r3", "B", sp.Integer(r3)),
    )


def thevenin(*, e1: int = 15, r3: int = 3) -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)` entre A et B, pour les `E1` et `R3` de l'enonce par defaut."""
    circuit = Reseau(_elements(e1, r3), "B")
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courant_dans_r2() -> sp.Expr:
    """`I1`, compte vers le haut : `-E2 = (R2 + R4) I1 + R4 I`."""
    return (-E2 - R4 * DEBIT) / (R2 + R4)


def tension_par_millman() -> sp.Expr:
    """Les courants de court-circuit des branches sur la somme des conductances."""
    return (DEBIT - E2 / R2) / (1 / R2 + 1 / R4)


def tension_sur_la_resistance() -> sp.Expr:
    """`U = -R' I2`, avec `E_TH - E' = (R_TH + R') I2`."""
    tension, resistance = thevenin()
    courant = (tension - PILE_DU_DIPOLE) / (resistance + RESISTANCE_DU_DIPOLE)
    return -RESISTANCE_DU_DIPOLE * courant


def tension_par_le_circuit() -> sp.Expr:
    """`U = V(B) - V(haut de R')`, le dipole branche sur le circuit d'origine."""
    dipole: tuple[Element, ...] = (
        SourceDeTension("A", "sur_rp", PILE_DU_DIPOLE),
        Resistance("sur_rp", "B", RESISTANCE_DU_DIPOLE),
    )
    potentiels = Reseau((*_elements(15, 3), *dipole), "B").potentiels()
    return sp.simplify(-potentiels["sur_rp"])
