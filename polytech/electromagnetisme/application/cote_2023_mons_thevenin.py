"""Exercice cote 2023 (Mons), question 1 : Thevenin en regime.

En regime la bobine `L` court-circuite `R2`, et le condensateur ouvre la
branche de `R3` : il reste `E0` et `R0`, puis `R1` et `R4`, entre A et B.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    resistance_vue_des_bornes,
    tension_a_vide,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Condensateur,
    Element,
    Inductance,
    Reseau,
    Resistance,
    SourceDeTension,
)

R0: Final = sp.Integer(1)
R1: Final = sp.Integer(1)
R4: Final = sp.Integer(4)
SOURCE: Final = sp.Integer(10)


def _elements(r2: int, r3: int) -> tuple[Element, ...]:
    return (
        SourceDeTension("sur_e", "B", SOURCE),
        Resistance("gauche", "sur_e", R0),
        Resistance("gauche", "B", R1),
        Inductance("gauche", "A"),
        Resistance("gauche", "A", sp.Integer(r2)),
        Condensateur("A", "sur_r3"),
        Resistance("sur_r3", "B", sp.Integer(r3)),
        Resistance("A", "B", R4),
    )


def thevenin(*, r2: int = 2, r3: int = 3) -> tuple[sp.Expr, sp.Expr]:
    """`(E_T, R_T)` entre A et B, pour les `R2` et `R3` de l'enonce par defaut."""
    circuit = Reseau(_elements(r2, r3), "B")
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def resistances_groupees() -> sp.Expr:
    """`R14`, le parallele de `R1` et `R4` que le manuscrit forme d'abord."""
    return R1 * R4 / (R1 + R4)


def tension_par_le_diviseur() -> sp.Expr:
    """`E_T = R14/(R0 + R14) E0` : un diviseur de tension."""
    groupe = resistances_groupees()
    return groupe / (R0 + groupe) * SOURCE


def court_circuit() -> sp.Expr:
    """`A` relie a `B` : `R1` et `R4` sont court-circuitees, il reste `E0/R0`."""
    return courant_de_court_circuit(Reseau(_elements(2, 3), "B"), "A", "B")
