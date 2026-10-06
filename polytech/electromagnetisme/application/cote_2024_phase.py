"""Exercice cote 2024 (Charleroi), question 3 : deux courants en phase.

Les branches `R1`-`L1` et `R2`-`L2` sont en parallele : elles voient la
meme tension, et le rapport de leurs courants est l'inverse de celui de
leurs impedances.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeTension,
)

R1, R2, L1, L2, W = sp.symbols("R_1 R_2 L_1 L_2 omega", positive=True)


def rapport() -> sp.Expr:
    """`I1/I2 = (R2 + j w L2)/(R1 + j w L1)` : la maille des deux branches."""
    return (R2 + sp.I * W * L2) / (R1 + sp.I * W * L1)


def dephasage() -> sp.Expr:
    """`arg(I1) - arg(I2)`, en radians."""
    return sp.atan(W * L2 / R2) - sp.atan(W * L1 / R1)


def condition() -> sp.Eq:
    """Ce qui annule la partie imaginaire du rapport, resolu en `L2`."""
    numerateur = sp.expand((R2 + sp.I * W * L2) * (R1 - sp.I * W * L1))
    (solution,) = sp.solve(sp.im(numerateur), L2)
    return sp.Eq(L2, solution)


def courants_du_reseau(  # noqa: PLR0913 -- les six parametres de la figure 3.1
    *, r0: int, r1: int, r2: int, l1: int, l2: int, w: int
) -> tuple[sp.Expr, sp.Expr]:
    """Les phaseurs de `i1` et `i2`, lus dans `R1` et `R2` du circuit entier."""
    circuit = Reseau(
        (
            SourceDeTension("s", "0", sp.Integer(1)),
            Resistance("s", "n", sp.Integer(r0)),
            Resistance("n", "a", sp.Integer(r1)),
            Resistance("a", "0", sp.I * w * l1),
            Resistance("n", "b", sp.Integer(r2)),
            Resistance("b", "0", sp.I * w * l2),
        ),
        "0",
    )
    potentiels = circuit.potentiels()
    premier = (potentiels["n"] - potentiels["a"]) / r1
    second = (potentiels["n"] - potentiels["b"]) / r2
    return sp.simplify(premier), sp.simplify(second)
