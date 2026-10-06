"""Exercice cote 2026 (Charleroi), question 1 : Thevenin a deux piles.

En regime la bobine est un fil : elle court-circuite `R3` et `R4`, et
tout le rail du haut est a la borne A. La branche `C`-`R5` est ouverte.
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
    SourceDeTension,
)

R1, R2, R3, R4, R5, R6 = (sp.Integer(valeur) for valeur in range(1, 7))
E1: Final = sp.Integer(10)
E2: Final = sp.Integer(5)


def _reseau(*, avec_la_bobine: bool, avec_la_branche_capacitive: bool) -> Reseau:
    elements: tuple[Element, ...] = (
        Resistance("A", "sur_e1", R1),
        SourceDeTension("sur_e1", "B", E1),
        Resistance("A", "entre", R3),
        Resistance("entre", "droite", R4),
        Resistance("droite", "sur_e2", R2),
        SourceDeTension("sur_e2", "B", E2),
        Resistance("droite", "B", R6),
    )
    if avec_la_bobine:
        elements = (*elements, Inductance("A", "droite"))
    if avec_la_branche_capacitive:
        branche = (Condensateur("droite", "sur_r5"), Resistance("sur_r5", "B", R5))
        elements = (*elements, *branche)
    return Reseau(elements, "B")


def thevenin(
    *, avec_la_bobine: bool = True, avec_la_branche_capacitive: bool = True
) -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)` entre A et B."""
    circuit = _reseau(
        avec_la_bobine=avec_la_bobine,
        avec_la_branche_capacitive=avec_la_branche_capacitive,
    )
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courants_de_maille() -> tuple[sp.Expr, sp.Expr]:
    """`(I1, I2)` du manuscrit : `I1` monte dans `E1`, `I2` descend dans `R2`."""
    premier, second = sp.symbols("I_1 I_2")
    mailles = (
        sp.Eq(E1 - E2, R1 * premier + R2 * second),
        sp.Eq(E2, -R2 * second + R6 * (premier - second)),
    )
    solution = sp.solve(mailles, (premier, second))
    return solution[premier], solution[second]


def tension_par_millman() -> sp.Expr:
    """La tension du noeud A : les courants de court-circuit sur les conductances."""
    return (E1 / R1 + E2 / R2) / (1 / R1 + 1 / R2 + 1 / R6)
