"""Exercice cote 2019, question 4 : Thevenin, deux sources de tension, une de courant.

En regime la bobine court-circuite `R2` et le condensateur est ouvert.
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
    Inductance,
    Reseau,
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)

R: Final = sp.Integer(1)
E: Final = sp.Integer(10)
DEBIT: Final = sp.Integer(1)


def reseau(r2: sp.Expr = R) -> Reseau:
    """La figure 4.1 entiere, borne B prise pour reference."""
    elements = (
        SourceDeCourant("B", "A", DEBIT),
        Resistance("A", "n", R),
        Resistance("n", "B", r2),
        Inductance("n", "B"),
        SourceDeTension("A", "m", E),
        Condensateur("A", "m"),
        Resistance("m", "B", R),
        Resistance("m", "k", R),
        SourceDeTension("k", "B", E),
    )
    return Reseau(elements, "B")


def thevenin(r2: sp.Expr = R) -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)` entre A et B."""
    circuit = reseau(r2)
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courant_de_court_circuit() -> sp.Expr:
    """Le courant dans un fil reliant A a B."""
    return court_circuit(reseau(), "A", "B")


def courants_de_mailles() -> tuple[sp.Expr, sp.Expr]:
    """`I1` et `I2` du manuscrit : ses deux mailles, resolues."""
    premier, second = sp.symbols("I_1 I_2")
    maille_une = sp.Eq(E, R * second - R * (premier - second))
    maille_deux = sp.Eq(E, R * (DEBIT + premier) + R * (premier - second))
    solution = sp.solve((maille_une, maille_deux), (premier, second))
    return solution[premier], solution[second]
