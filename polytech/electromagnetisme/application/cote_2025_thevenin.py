"""Exercice cote 2025 (Charleroi), question 1 : Thevenin, puis un dipole actif.

En regime la bobine est un fil et la branche `C`-`R4` est ouverte. Les
deux piles ont leur pole positif du cote de leur resistance.
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

R1, R2, R3, R4 = (sp.Integer(valeur) for valeur in (1, 2, 3, 4))
E1: Final = sp.Integer(15)
E2: Final = sp.Integer(10)
DEBIT: Final = sp.Integer(2)
CHARGE: Final = sp.Integer(6)
"""`R'`, en ohms."""
DEBIT_DU_DIPOLE: Final = sp.Integer(2)
"""`I'`, en amperes, injecte dans la borne A."""


def _elements(*, avec_la_branche_capacitive: bool) -> tuple[Element, ...]:
    elements: tuple[Element, ...] = (
        SourceDeCourant("B", "haut", DEBIT),
        SourceDeTension("sous_e2", "haut", E2),
        Resistance("sous_e2", "B", R2),
        Resistance("haut", "avant_e1", R1),
        SourceDeTension("avant_e1", "A", E1),
        Inductance("A", "sur_r3"),
        Resistance("sur_r3", "B", R3),
    )
    if avec_la_branche_capacitive:
        branche = (Condensateur("haut", "sur_r4"), Resistance("sur_r4", "B", R4))
        elements = (*elements, *branche)
    return elements


def thevenin(*, avec_la_branche_capacitive: bool = True) -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)`, `E_TH` etant `V_A - V_B`."""
    circuit = Reseau(
        _elements(avec_la_branche_capacitive=avec_la_branche_capacitive), "B"
    )
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courant_de_maille() -> sp.Expr:
    """`I1`, de `R1` vers `R3` : la loi des mailles du manuscrit."""
    return (-E2 - E1 + R2 * DEBIT) / (R1 + R2 + R3)


def tension_par_l_equivalent() -> sp.Expr:
    """`U = R' I2`, le dipole branche sur le generateur de Thevenin."""
    tension, resistance = thevenin()
    courant = (tension + resistance * DEBIT_DU_DIPOLE) / (resistance + CHARGE)
    return CHARGE * courant


def tension_par_le_circuit() -> sp.Expr:
    """`U = V_A - V_B`, le dipole branche sur le circuit d'origine."""
    dipole: tuple[Element, ...] = (
        Resistance("A", "B", CHARGE),
        SourceDeCourant("B", "A", DEBIT_DU_DIPOLE),
    )
    circuit = Reseau((*_elements(avec_la_branche_capacitive=True), *dipole), "B")
    return circuit.potentiels()["A"]
