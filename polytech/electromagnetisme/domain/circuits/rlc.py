"""Le circuit RLC serie, ecrit comme une equation differentielle.

La tension `u` appliquee se partage entre la bobine, la resistance et le
condensateur (loi des mailles) :

    u = L di/dt + R i + (1/C) int_0^t i dt.

En derivant, l'integrale disparait et l'on obtient une equation
differentielle lineaire du second ordre a coefficients constants :

    du/dt = L d2i/dt2 + R di/dt + i / C.

C'est l'exemple par lequel le cours de mathematique introduit les
equations differentielles ; il vit ici, avec les circuits.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

TEMPS: Final = sp.Symbol("t", nonnegative=True)
COURANT: Final = sp.Function("i")
TENSION: Final = sp.Function("u")
_MUETTE: Final = sp.Symbol("tau", nonnegative=True)


def equation_integrale_du_circuit_rlc(
    resistance: object,
    inductance: object,
    capacite: object,
) -> sp.Eq:
    """`u = L di/dt + R i + (1/C) int_0^t i dt`."""
    courant = COURANT(TEMPS)
    charge = sp.Integral(COURANT(_MUETTE), (_MUETTE, 0, TEMPS))
    membre = (
        sp.sympify(inductance) * courant.diff(TEMPS)
        + sp.sympify(resistance) * courant
        + charge / sp.sympify(capacite)
    )
    return sp.Eq(TENSION(TEMPS), membre)


def equation_du_circuit_rlc(
    resistance: object,
    inductance: object,
    capacite: object,
) -> sp.Eq:
    """`du/dt = L i'' + R i' + i / C`."""
    courant = COURANT(TEMPS)
    membre = (
        sp.sympify(inductance) * courant.diff(TEMPS, 2)
        + sp.sympify(resistance) * courant.diff(TEMPS)
        + courant / sp.sympify(capacite)
    )
    return sp.Eq(TENSION(TEMPS).diff(TEMPS), membre)
