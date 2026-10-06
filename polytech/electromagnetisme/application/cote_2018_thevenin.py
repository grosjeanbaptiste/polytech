"""Exercice cote 2018, question 3 : Thevenin avec une source de courant.

En regime, la branche `L1`-`C1` et la branche `C3`-`R3` sont ouvertes :
il reste `E` et `R1`, la source de courant et `R2`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.equivalents import (
    eteindre,
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

R: Final = sp.Integer(5)
"""`R1 = R2 = R3 = R' = 5` ohms."""
E: Final = sp.Integer(10)
DEBIT: Final = sp.Integer(1)
"""`I = 1` A, le courant de la source de courant."""
_TENSION: Final = SourceDeTension("s", "B", E)
_COURANT: Final = SourceDeCourant("B", "A", DEBIT)


def reseau() -> Reseau:
    """La figure 3.1(a) entiere, borne B prise pour reference."""
    elements = (
        _TENSION,
        Resistance("s", "A", R),
        Inductance("A", "m"),
        Condensateur("m", "B"),
        _COURANT,
        Resistance("A", "B", R),
        Condensateur("A", "n"),
        Resistance("n", "B", R),
    )
    return Reseau(elements, "B")


def thevenin() -> tuple[sp.Expr, sp.Expr]:
    """`(E_TH, R_TH)` entre A et B."""
    circuit = reseau()
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def tension_par_millman() -> sp.Expr:
    """`(E/R1 + I) / (1/R1 + 1/R2)` : la source de courant s'ajoute au numerateur."""
    return (E / R + DEBIT) / (1 / R + 1 / R)


def tension_par_superposition() -> tuple[sp.Expr, sp.Expr]:
    """La part de `E` seule, puis celle de `I` seule."""
    parts = []
    for gardee in (_TENSION, _COURANT):
        eteint = eteindre(reseau())
        rang = reseau().elements.index(gardee)
        elements = (*eteint.elements[:rang], gardee, *eteint.elements[rang + 1 :])
        parts.append(tension_a_vide(replace(eteint, elements=elements), "A", "B"))
    return parts[0], parts[1]


def courant_de_charge() -> sp.Expr:
    """`I = E_TH / (R_TH + R')`, la bobine `L'` etant un fil en regime."""
    tension, resistance = thevenin()
    return tension / (resistance + R)


def courant_sur_le_reseau_complet() -> sp.Expr:
    """Le dipole `L'`-`R'` branche sur le circuit d'origine, sans equivalent."""
    complet = reseau()
    dipole = (Inductance("A", "q"), Resistance("q", "B", R))
    charge = replace(complet, elements=(*complet.elements, *dipole))
    return sp.simplify(charge.potentiels()["q"] / R)
