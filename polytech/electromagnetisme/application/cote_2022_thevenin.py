"""Exercice cote 2022 (Charleroi), question 2 : Thevenin, puis un condensateur.

En regime la bobine est un fil et la branche `C`-`R2` est ouverte.
"""

from __future__ import annotations

from dataclasses import replace
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
from polytech.electromagnetisme.domain.energie_et_forces import (
    energie_d_un_condensateur,
)

R: Final = sp.Integer(1)
CHARGE: Final = sp.Integer(5)
"""`R0 = 5` ohms."""
E: Final = sp.Integer(10)
DEBIT: Final = sp.Integer(2)
CAPACITE: Final = sp.Rational(1, 10**6)


def reseau() -> Reseau:
    """La figure 2.1(a) entiere, borne B (rail du bas) prise pour reference."""
    elements = (
        SourceDeTension("s", "B", E),
        Resistance("s", "g", R),
        Inductance("g", "h"),
        Condensateur("h", "c"),
        Resistance("c", "B", R),
        Resistance("h", "A", R),
        Resistance("A", "B", R),
        SourceDeCourant("B", "h", DEBIT),
    )
    return Reseau(elements, "B")


def thevenin() -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)` entre A et B."""
    circuit = reseau()
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def courant_dans_r4() -> sp.Expr:
    """`I4 = (E + R1 I)/(R1 + R3 + R4)`, la maille du manuscrit."""
    return (E + R * DEBIT) / (3 * R)


def courant_de_court_circuit() -> sp.Expr:
    """Le courant dans un fil reliant A a B."""
    return court_circuit(reseau(), "A", "B")


def charge_du_dipole() -> tuple[sp.Expr, sp.Expr]:
    """La tension de `C0` en regime (diviseur sur `R0`), puis son energie."""
    tension, resistance = thevenin()
    aux_bornes = CHARGE / (resistance + CHARGE) * tension
    return aux_bornes, energie_d_un_condensateur(CAPACITE, aux_bornes)


def tension_sur_le_reseau_complet() -> sp.Expr:
    """`C0` et `R0` branches sur le circuit d'origine, sans equivalent."""
    complet = reseau()
    dipole = (Condensateur("A", "B"), Resistance("A", "B", CHARGE))
    charge = replace(complet, elements=(*complet.elements, *dipole))
    return sp.simplify(charge.potentiels()["A"])
