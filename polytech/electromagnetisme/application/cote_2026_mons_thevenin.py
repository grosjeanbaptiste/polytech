"""Exercice cote 2026 (Mons), question 1 : Thevenin en regime.

En regime `C` ouvre la branche de `R4`, et `L` court-circuite `R5` : il
reste la branche `E1`-`R1`, aux bornes de laquelle sont A et B, puis `R3`,
la branche `R2`-`E2` et `R6`.
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

E1: Final = sp.Integer(10)
E2: Final = sp.Integer(5)
R1: Final = sp.Integer(1)
R2: Final = sp.Integer(2)
R3: Final = sp.Integer(3)
R6: Final = sp.Integer(6)


def _circuit(r4: int, r5: int) -> Reseau:
    elements: tuple[Element, ...] = (
        SourceDeTension("sur_e1", "B", E1),
        Resistance("sur_e1", "A", R1),
        Resistance("A", "droite", R3),
        Resistance("droite", "sur_e2", R2),
        SourceDeTension("sur_e2", "B", E2),
        Condensateur("droite", "sur_r4"),
        Resistance("sur_r4", "B", sp.Integer(r4)),
        Resistance("droite", "sur_r6", sp.Integer(r5)),
        Inductance("droite", "sur_r6"),
        Resistance("sur_r6", "B", R6),
    )
    return Reseau(elements, "B")


def thevenin(*, r4: int = 4, r5: int = 5) -> tuple[sp.Expr, sp.Expr]:
    """`(E_T, R_T)` entre A et B, pour les `R4` et `R5` de l'enonce par defaut."""
    circuit = _circuit(r4, r5)
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courants_de_maille() -> tuple[sp.Expr, sp.Expr]:
    """`(I1, I2)` du manuscrit : `I1` dans `R1` et `R3`, `I2` descend dans `R2`."""
    premier, second = sp.symbols("I_1 I_2")
    mailles = (
        sp.Eq(E1 - E2, R1 * premier + R3 * premier + R2 * second),
        sp.Eq(E2, -R2 * second + R6 * (premier - second)),
    )
    solution = sp.solve(mailles, (premier, second))
    return solution[premier], solution[second]


def court_circuit() -> sp.Expr:
    """Le courant de A vers B dans un fil qui relie les deux bornes."""
    return courant_de_court_circuit(_circuit(4, 5), "A", "B")
