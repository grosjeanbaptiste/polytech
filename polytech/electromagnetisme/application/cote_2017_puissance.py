"""Exercice cote 2017, question 2 : trois mailles couplees, puis une puissance.

La premiere partie est litterale ; la seconde chiffre la branche
`C2`-`R2`-`e2` avec les courants que l'enonce donne.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Final

import sympy as sp

R1, R2, R4 = sp.symbols("R_1 R_2 R_4", positive=True)
L1, L3, L4, L34 = sp.symbols("L_1 L_3 L_4 L_34", positive=True)
C2, W = sp.symbols("C_2 omega", positive=True)
E1, E2, E4 = sp.symbols("E_1 E_2 E_4")

TENSION_EFFICACE: Final = 220.0
RETARD_DE_E2_EN_DEGRES: Final = 10.0
RESISTANCE: Final = 10.0
CAPACITE: Final = 10e-6
PULSATION: Final = 2 * math.pi * 50.0
I1_DONNE: Final = complex(1, 1)
I3_DONNE: Final = complex(3, 3)
"""Les courants que l'enonce donne, en amperes."""


def systeme() -> tuple[sp.Matrix, sp.Matrix]:
    """La matrice des mailles et son second membre, inconnues `I1`, `I3`, `I4`."""
    mediane = R2 + 1 / (sp.I * W * C2)
    mutuelle = -sp.I * W * L34
    matrice = sp.Matrix(
        [
            [R1 + sp.I * W * L1 + mediane, -mediane, 0],
            [-mediane, mediane + sp.I * W * L3, mutuelle],
            [0, mutuelle, R4 + sp.I * W * L4],
        ]
    )
    return matrice, sp.Matrix([E1 - E2, E2, E4])


def _source() -> complex:
    return cmath.rect(TENSION_EFFICACE, -math.radians(RETARD_DE_E2_EN_DEGRES))


def _impedance() -> complex:
    return RESISTANCE + 1 / (1j * PULSATION * CAPACITE)


def courant_de_la_branche() -> complex:
    """`I12 = I1 - I3`, de haut en bas."""
    return I1_DONNE - I3_DONNE


def tension_de_la_branche() -> complex:
    """`U12 = E2 + (R2 + 1/(j omega C2)) I12`."""
    return _source() + _impedance() * courant_de_la_branche()


def puissance_de_la_branche() -> complex:
    """`S12 = U12 I12*`, convention recepteur."""
    return tension_de_la_branche() * courant_de_la_branche().conjugate()


@dataclass(frozen=True, slots=True)
class Bilan:
    """Ce que chaque element de la branche absorbe, en watts ou en vars."""

    source_active: float
    source_reactive: float
    resistance: float
    condensateur: float


def bilan_par_element() -> Bilan:
    """La source, puis `R2 |I|^2`, puis `-|I|^2 / (omega C2)`."""
    courant = courant_de_la_branche()
    de_la_source = _source() * courant.conjugate()
    carre = abs(courant) ** 2
    return Bilan(
        de_la_source.real,
        de_la_source.imag,
        RESISTANCE * carre,
        -carre / (PULSATION * CAPACITE),
    )
