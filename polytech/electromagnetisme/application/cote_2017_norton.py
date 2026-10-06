"""Exercice cote 2017, question 1 : Norton en regime, puis un dipole.

Toutes les resistances valent 1 ohm et `E` 10 V. En regime continu les
bobines sont des fils et les condensateurs des circuits ouverts : le
condensateur du rail isole toute la partie droite du circuit, et la
branche condensateur-resistance de gauche ne porte aucun courant.
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
)
from polytech.electromagnetisme.domain.circuits.equivalents import (
    tension_a_vide as a_vide,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Condensateur,
    Inductance,
    Reseau,
    Resistance,
    SourceDeTension,
)

R: Final = sp.Integer(1)
E: Final = sp.Integer(10)


def reseau(r: sp.Expr = R) -> Reseau:
    """La figure 1.1 entiere, rail du bas (borne B) pris pour reference."""
    elements = (
        Inductance("t1", "g"),
        Resistance("g", "B", r),
        Resistance("t1", "t2", r),
        Condensateur("t2", "c"),
        Resistance("c", "B", r),
        Resistance("t2", "t3", r),
        SourceDeTension("t3", "e", E),
        Resistance("e", "B", r),
        Resistance("t3", "A", r),
        Condensateur("t3", "t4"),
        Inductance("t4", "d"),
        Resistance("d", "B", r),
        Resistance("t4", "B", r),
    )
    return Reseau(elements, "B")


def tension_a_vide(r: sp.Expr = R) -> sp.Expr:
    """`E_TH`, la tension entre A et B sans rien y brancher."""
    return a_vide(reseau(r), "A", "B")


def courant_de_court_circuit() -> sp.Expr:
    """Le courant dans un fil reliant A a B, mesure sur le reseau."""
    return court_circuit(reseau(), "A", "B")


def norton(r: sp.Expr = R) -> tuple[sp.Expr, sp.Expr]:
    """`(I_N, R_N)`, le courant etant obtenu comme le manuscrit : `E_TH / R_N`."""
    resistance = resistance_vue_des_bornes(reseau(r), "A", "B")
    return tension_a_vide(r) / resistance, resistance


def resistance_en_parallele(r: sp.Expr = R) -> sp.Expr:
    """`R' = R_N // 2R`, ce que le courant `I_N - I` traverse."""
    _, resistance = norton(r)
    return 2 * r * resistance / (2 * r + resistance)


def courant_du_dipole(r: sp.Expr = R) -> sp.Expr:
    """`I = R' I_N / (R + R')` : la maille du manuscrit."""
    courant, _ = norton(r)
    autre = resistance_en_parallele(r)
    return autre * courant / (r + autre)


def courant_par_le_diviseur(r: sp.Expr = R) -> sp.Expr:
    """`I = I_N G / (G_N + G + G/2)` : la part du courant qui prend la branche `R`."""
    courant, resistance = norton(r)
    conductances = 1 / resistance + 1 / r + 1 / (2 * r)
    return courant * (1 / r) / conductances


def courant_sur_le_reseau_complet(r: sp.Expr = R) -> sp.Expr:
    """Le dipole de la figure 1.2 branche sur le circuit d'origine."""
    dipole = (
        Resistance("A", "B", r),
        Resistance("A", "m", r),
        Resistance("m", "B", r),
    )
    complet = reseau(r)
    charge = replace(complet, elements=(*complet.elements, *dipole))
    return sp.simplify(charge.potentiels()["A"] / r)
