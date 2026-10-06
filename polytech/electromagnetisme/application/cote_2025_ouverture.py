"""Exercice cote 2025 (Charleroi), question 3 : ouvrir d'un cote, fermer de l'autre.

Avant `t = 0` la bobine est en serie avec la source, `R0` et `R1` ;
apres, elle se referme sur `R1` et `R2`. Le courant `i` est compte **vers
le bas** dans la bobine, avant comme apres : c'est lui qui ne saute pas.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.phaseurs import valeur_instantanee
from polytech.electromagnetisme.domain.circuits.reseau import (
    Reseau,
    Resistance,
    SourceDeCourant,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

T: Final = TEMPS
E0, R0, R1, R2, L, W = sp.symbols("E_0 R_0 R_1 R_2 L omega", positive=True)


def courant_avant() -> sp.Expr:
    """`i(t)` pour `t < 0` : `I = E / (R0 + R1 + j omega L)`."""
    return valeur_instantanee(E0 / (R0 + R1 + sp.I * W * L), W)


def courant_initial() -> sp.Expr:
    """`i(0)`, que la bobine conserve a la commutation."""
    return sp.factor(sp.simplify(courant_avant().subs(T, 0)))


def courant_apres() -> sp.Expr:
    """`i(t)` pour `t > 0`, toujours compte vers le bas dans la bobine."""
    return courant_initial() * sp.exp(-(R1 + R2) * T / L)


def tension() -> sp.Expr:
    """`u(t)`, flechee vers le haut sur `R2` : le courant y **remonte**."""
    return -R2 * courant_apres()


def tension_du_manuscrit() -> sp.Expr:
    """Ce que le manuscrit ecrit : `u = +R2 i(0) e^(-(R1+R2)t/L)`."""
    return R2 * courant_apres()


def tension_initiale_par_le_reseau() -> sp.Expr:
    """`u(0+)` : la bobine remplacee par une source du courant qu'elle porte."""
    circuit = Reseau(
        (
            Resistance("haut", "milieu", R1),
            SourceDeCourant("milieu", "bas", courant_initial()),
            Resistance("haut", "bas", R2),
        ),
        "bas",
    )
    return circuit.potentiels()["haut"]
