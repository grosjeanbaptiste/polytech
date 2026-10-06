"""Exercice cote 2024 (Charleroi), question 1 : Thevenin, puis un condensateur.

En regime la bobine est un fil : elle court-circuite `R4`, et la borne A
se confond avec le noeud du haut.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import sympy as sp

from polytech.electromagnetisme.domain.circuits.equivalents import (
    resistance_vue_des_bornes,
    tension_a_vide,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Element,
    Inductance,
    Reseau,
    Resistance,
    SourceDeTension,
)

SOURCE: Final = sp.Integer(10)
CAPACITE: Final = 2e-6
"""`C0`, en farads."""


class Resistances(NamedTuple):
    """`R1` a `R4`, en ohms."""

    premiere: sp.Expr
    deuxieme: sp.Expr
    troisieme: sp.Expr
    quatrieme: sp.Expr


ENONCE: Final = Resistances(*(sp.Integer(valeur) for valeur in (1, 2, 3, 4)))


def reseau(resistances: Resistances, *, avec_la_bobine: bool = True) -> Reseau:
    """La figure 1.1(a) ; le pole positif de `E` est du cote de `R3`."""
    r1, r2, r3, r4 = resistances
    elements: tuple[Element, ...] = (
        Resistance("haut", "A", r4),
        Resistance("haut", "coin", r1),
        Resistance("coin", "B", r2),
        SourceDeTension("pile", "A", SOURCE),
        Resistance("pile", "B", r3),
    )
    if avec_la_bobine:
        elements = (Inductance("haut", "A"), *elements)
    return Reseau(elements, "A")


def thevenin(
    *, resistances: Resistances = ENONCE, avec_la_bobine: bool = True
) -> tuple[sp.Expr, sp.Expr]:
    """`(E_T, R_T)`, `E_T` etant `V_A - V_B` comme le manuscrit la fleche."""
    circuit = reseau(resistances, avec_la_bobine=avec_la_bobine)
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def tension_par_le_diviseur(resistances: Resistances = ENONCE) -> sp.Expr:
    """`V_B - V_A` : la part de `E` que prennent `R1 + R2`, en serie avec `R3`."""
    r1, r2, r3, _ = resistances
    return SOURCE * (r1 + r2) / (r1 + r2 + r3)


def energie() -> float:
    """`1/2 C0 E_T^2`, en joules : en regime `C0` ne debite rien et prend `E_T`."""
    tension, _ = thevenin()
    return 0.5 * CAPACITE * float(tension) ** 2
