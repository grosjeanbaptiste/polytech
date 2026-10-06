"""Exercice cote 2023 (Charleroi), question 1 : Thevenin autour d'une source de courant.

En regime la bobine est un fil et le condensateur, en parallele sur
`R2`, est ouvert.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit as court_circuit,
)
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
)

R1, R2, R3, R4 = (sp.Integer(valeur) for valeur in (1, 2, 3, 4))
DEBIT: Final = sp.Integer(5)


def reseau(*, avec_le_condensateur: bool = True) -> Reseau:
    """La figure 1.1, borne B (rail du bas) prise pour reference."""
    elements: tuple[Element, ...] = (
        SourceDeCourant("B", "g", DEBIT),
        Resistance("g", "B", R1),
        Resistance("g", "A", R2),
        Inductance("A", "m"),
        Resistance("m", "B", R3),
        Resistance("A", "B", R4),
    )
    if avec_le_condensateur:
        elements = (*elements, Condensateur("g", "A"))
    return Reseau(elements, "B")


def thevenin(*, avec_le_condensateur: bool = True) -> tuple[sp.Expr, sp.Expr]:
    """`(E_T, R_T)` entre A et B."""
    circuit = reseau(avec_le_condensateur=avec_le_condensateur)
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courant_dans_r2() -> sp.Expr:
    """`I1 = R1 I / (R1 + R2 + R3 // R4)` : le diviseur de courant du manuscrit."""
    parallele = R3 * R4 / (R3 + R4)
    return R1 * DEBIT / (R1 + R2 + parallele)


def courant_de_court_circuit() -> sp.Expr:
    """Le courant dans un fil reliant A a B."""
    return court_circuit(reseau(), "A", "B")
