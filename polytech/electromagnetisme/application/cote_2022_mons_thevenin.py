"""Exercice cote 2022 (Mons), question 2 : Thevenin, puis une bobine reelle.

En regime `L1` et `L2` sont des fils, et la branche `C`-`R2` est ouverte :
il reste trois branches en parallele entre A et B.
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

UN_OHM: Final = sp.Integer(1)
"""`R1` a `R4`, toutes egales."""
SOURCE: Final = sp.Integer(10)
DEBIT: Final = sp.Integer(2)
RESISTANCE_DE_LA_BOBINE: Final = sp.Rational(1, 10)


def _elements() -> tuple[Element, ...]:
    return (
        SourceDeCourant("B", "gauche", DEBIT),
        Resistance("gauche", "B", UN_OHM),
        Inductance("gauche", "droite"),
        Condensateur("droite", "sur_r2"),
        Resistance("sur_r2", "B", UN_OHM),
        Inductance("droite", "A"),
        Resistance("A", "B", UN_OHM),
        Resistance("droite", "sur_e", UN_OHM),
        SourceDeTension("sur_e", "B", SOURCE),
    )


def thevenin() -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)` entre A et B."""
    circuit = Reseau(_elements(), "B")
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def tension_par_millman() -> sp.Expr:
    """Trois branches en parallele : courants de court-circuit sur conductances."""
    return (DEBIT + SOURCE / UN_OHM) / (3 / UN_OHM)


def courants_de_maille() -> tuple[sp.Expr, sp.Expr]:
    """`(I3, I4)` du manuscrit : dans `R3` vers le bas, dans `R4` vers le bas."""
    trois, quatre = sp.symbols("I_3 I_4")
    mailles = (
        sp.Eq(0, UN_OHM * (trois + quatre - DEBIT) + UN_OHM * trois),
        sp.Eq(-SOURCE, -UN_OHM * trois + UN_OHM * quatre),
    )
    solution = sp.solve(mailles, (trois, quatre))
    return solution[trois], solution[quatre]


def bobine() -> tuple[sp.Expr, sp.Expr]:
    """Le courant de regime dans la bobine, et la puissance qu'elle dissipe."""
    tension, resistance = thevenin()
    courant = tension / (resistance + RESISTANCE_DE_LA_BOBINE)
    return courant, RESISTANCE_DE_LA_BOBINE * courant**2


def courant_par_le_circuit() -> sp.Expr:
    """Le meme courant, la bobine branchee sur le circuit d'origine."""
    reelle: tuple[Element, ...] = (
        Inductance("A", "dans_la_bobine"),
        Resistance("dans_la_bobine", "B", RESISTANCE_DE_LA_BOBINE),
    )
    circuit = Reseau((*_elements(), *reelle), "B")
    return sp.simplify(circuit.potentiels()["dans_la_bobine"] / RESISTANCE_DE_LA_BOBINE)
