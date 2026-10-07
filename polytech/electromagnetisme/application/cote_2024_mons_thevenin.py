"""Exercice cote 2024 (Mons), question 1 : Thevenin, puis l'energie d'une capacite.

En regime `L` est un fil et `C` ouvre la branche de `R5` : il reste une
seule maille, `E`, `R3`, `R2`, `R1`, `R4`. La plaque longue de la pile,
son pole positif, est du cote de `R3`.
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

R1: Final = sp.Integer(1)
R2: Final = sp.Integer(2)
R3: Final = sp.Integer(3)
R4: Final = sp.Integer(4)
SOURCE: Final = sp.Integer(10)
CAPACITE_EN_MICROFARADS: Final = sp.Integer(2)
"""`C0`, la capacite que l'on branche entre A et B."""


def _elements(r5: int) -> tuple[Element, ...]:
    return (
        Condensateur("gauche", "sur_r5"),
        Resistance("sur_r5", "bas", sp.Integer(r5)),
        Resistance("gauche", "bas", R4),
        Resistance("gauche", "apres_r1", R1),
        Inductance("apres_r1", "A"),
        Resistance("A", "B", R2),
        Resistance("B", "sur_e", R3),
        SourceDeTension("sur_e", "bas", SOURCE),
    )


def thevenin(*, r5: int = 5) -> tuple[sp.Expr, sp.Expr]:
    """`(E_T, R_T)` entre A et B, pour le `R5` de l'enonce par defaut."""
    circuit = Reseau(_elements(r5), "B")
    return tension_a_vide(circuit, "A", "B"), resistance_vue_des_bornes(
        circuit, "A", "B"
    )


def resistance_par_la_formule() -> sp.Expr:
    """`(R1 + R4 + R3) // R2`, comme le manuscrit l'ecrit."""
    return (R1 * R2 + R2 * R3 + R2 * R4) / (R1 + R2 + R3 + R4)


def courant_de_la_maille() -> sp.Expr:
    """`I`, compte de A vers B dans `R2` : `-E = (R1 + R2 + R3 + R4) I`."""
    return -SOURCE / (R1 + R2 + R3 + R4)


def energie_en_microjoules() -> sp.Expr:
    """`(1/2) C0 E_T^2` : en regime la capacite porte la tension a vide."""
    tension, _ = thevenin()
    return CAPACITE_EN_MICROFARADS * tension**2 / 2


def tension_de_la_capacite() -> sp.Expr:
    """`V(A) - V(B)`, la capacite `C0` branchee sur le circuit d'origine."""
    charge: tuple[Element, ...] = (Condensateur("A", "B"),)
    return tension_a_vide(Reseau((*_elements(5), *charge), "B"), "A", "B")
