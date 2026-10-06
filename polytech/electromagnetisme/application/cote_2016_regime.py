"""Exercice cote 2016, question 1 : Thevenin en regime, puis un condensateur.

En regime continu une inductance est un fil et un condensateur un circuit
ouvert : la branche `E2`-`C2` ne porte aucun courant, et le reseau se
reduit a trois branches en parallele entre A et B.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import cote_2016_enonce as en
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
    SourceDeTension,
)
from polytech.electromagnetisme.domain.energie_et_forces import (
    energie_d_un_condensateur,
)


def reseau(e2: sp.Expr | None = None) -> Reseau:
    """La figure 1.1(a), noeud B pris pour reference."""
    seconde = sp.Integer(en.E2) if e2 is None else e2
    elements = (
        SourceDeTension("a1", "b1", sp.Integer(en.E1)),
        Resistance("b1", "B", sp.Integer(en.R1)),
        Inductance("a1", "A"),
        SourceDeTension("A", "c2", seconde),
        Condensateur("c2", "B"),
        SourceDeTension("A", "a3", sp.Integer(en.E3)),
        Resistance("a3", "l3", sp.Integer(en.R3_Q1)),
        Inductance("l3", "B"),
        Resistance("A", "B", sp.Integer(en.R4)),
    )
    return Reseau(elements, "B")


def thevenin(e2: sp.Expr | None = None) -> tuple[sp.Expr, sp.Expr]:
    """`(E0, R0)` entre A et B."""
    circuit = reseau(e2)
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def tension_par_millman() -> sp.Expr:
    """`V_A = somme(E_k / R_k) / somme(1 / R_k)` sur les trois branches actives."""
    conductances = [sp.Rational(1, r) for r in (en.R1, en.R3_Q1, en.R4)]
    courants = [sp.Rational(en.E1, en.R1), sp.Rational(en.E3, en.R3_Q1), 0]
    return sp.Add(*courants) / sp.Add(*conductances)


def courant_de_court_circuit() -> sp.Expr:
    """Le courant dans un fil reliant A a B."""
    return court_circuit(reseau(), "A", "B")


def charge_du_dipole() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le courant dans `R`, la tension de `C` en regime, l'energie emmagasinee."""
    tension, resistance = thevenin()
    courant = tension / (resistance + en.CHARGE_R)
    aux_bornes = en.CHARGE_R * courant
    capacite = sp.Rational(en.CHARGE_C_EN_MICROFARADS, 10**6)
    return courant, aux_bornes, energie_d_un_condensateur(capacite, aux_bornes)


def constante_de_charge() -> sp.Expr:
    """`(R0 // R) C`, en secondes : le temps qu'il faut pour atteindre ce regime."""
    _, resistance = thevenin()
    vue = resistance * en.CHARGE_R / (resistance + en.CHARGE_R)
    return vue * sp.Rational(en.CHARGE_C_EN_MICROFARADS, 10**6)
