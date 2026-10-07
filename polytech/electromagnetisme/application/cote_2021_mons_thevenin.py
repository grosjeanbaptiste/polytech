"""Exercice cote 2021 (Mons), question 2 : Thevenin, puis une bobine resistive.

En regime `L1` est un fil, `C1` et `C2` sont ouverts : la branche `R5`-`C2`
ne porte aucun courant.
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

UN_OHM: Final = sp.Integer(1)
"""`R1` a `R5`, toutes egales."""
DEBIT: Final = sp.Integer(3)
CHARGE: Final = sp.Integer(5)
"""`R0`, la resistance du dipole."""


def _elements() -> tuple[Element, ...]:
    return (
        SourceDeCourant("B", "haut", DEBIT),
        Resistance("haut", "B", UN_OHM),
        Inductance("haut", "sous_l1"),
        Resistance("sous_l1", "B", UN_OHM),
        Resistance("haut", "B", UN_OHM),
        Resistance("haut", "A", UN_OHM),
        Condensateur("haut", "A"),
        Resistance("A", "sur_c2", UN_OHM),
        Condensateur("sur_c2", "B"),
    )


def thevenin() -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)` entre A et B."""
    circuit = Reseau(_elements(), "B")
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courant_de_court_circuit() -> sp.Expr:
    """Le courant dans un fil reliant A a B."""
    return court_circuit(Reseau(_elements(), "B"), "A", "B")


def courant_par_l_equivalent() -> sp.Expr:
    """`I' = E_TH / (R_TH + R0)` : en regime la bobine `L0` est un fil."""
    tension, resistance = thevenin()
    return tension / (resistance + CHARGE)


def courant_par_le_circuit() -> sp.Expr:
    """Le meme courant, le dipole branche sur le circuit d'origine."""
    dipole: tuple[Element, ...] = (
        Inductance("A", "sous_l0"),
        Resistance("sous_l0", "B", CHARGE),
    )
    circuit = Reseau((*_elements(), *dipole), "B")
    return sp.simplify(circuit.potentiels()["sous_l0"] / CHARGE)
