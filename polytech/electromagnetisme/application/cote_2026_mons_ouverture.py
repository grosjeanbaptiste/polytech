"""Exercice cote 2026 (Mons), question 3 : l'ouverture d'un interrupteur.

Ferme, l'interrupteur met la branche `R`-`L` et le condensateur sous la
source `E1`, a travers `R1`. Ouvert, il laisse la maille `R`, `L`, `C`
seule. `i` monte dans `R` ; la fleche de `u_c` pointe vers le bas.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.reseau import (
    Condensateur,
    Inductance,
    Reseau,
    Resistance,
    SourceDeTension,
)

E1, R, R1, L, C, P = sp.symbols("E_1 R R_1 L C p", positive=True)
TRANSFORMEES: Final = sp.symbols("I U_c")


def avant() -> tuple[sp.Expr, sp.Expr]:
    """`(i, u_c)` juste avant : le courant descend dans `R`, a rebours de `i`."""
    courant = -E1 / (R1 + R)
    return courant, R * courant


def avant_par_le_reseau() -> tuple[sp.Expr, sp.Expr]:
    """Les memes, le circuit ferme resolu comme un reseau en regime."""
    circuit = Reseau(
        (
            SourceDeTension("sur_e", "bas", E1),
            Resistance("sur_e", "haut", R1),
            Resistance("haut", "milieu", R),
            Inductance("milieu", "bas"),
            Condensateur("haut", "bas"),
        ),
        "bas",
    )
    potentiels = circuit.potentiels()
    courant = (potentiels["milieu"] - potentiels["haut"]) / R
    return courant, -potentiels["haut"]


def transformee() -> sp.Expr:
    """`I(p)` : `u_c = R i + L i'` et `i = -C u_c'`, conditions initiales comprises."""
    courant, tension = TRANSFORMEES
    depart_i, depart_u = avant()
    equations = (
        tension - R * courant - L * (P * courant - depart_i),
        courant + C * (P * tension - depart_u),
    )
    return sp.simplify(sp.solve(equations, TRANSFORMEES)[courant])


def polynomes() -> tuple[sp.Expr, sp.Expr]:
    """Le numerateur et le denominateur de `I(p)`, developpes en `p`."""
    numerateur, denominateur = sp.fraction(sp.together(transformee()))
    return sp.expand(numerateur), sp.expand(denominateur)


def pente_initiale() -> sp.Expr:
    """`i'(0) = (u_c(0) - R i(0)) / L` : la maille, ecrite a l'instant zero."""
    courant, tension = avant()
    return sp.simplify((tension - R * courant) / L)


def transformee_par_l_equation() -> sp.Expr:
    """`I(p)` tire de `LC i'' + RC i' + i = 0`, avec `i(0)` et `i'(0)`."""
    depart, _ = avant()
    numerateur = L * C * (P * depart + pente_initiale()) + R * C * depart
    return numerateur / (L * C * P**2 + R * C * P + 1)


def limites() -> tuple[sp.Expr, sp.Expr]:
    """`lim p I(p)` a l'infini (la valeur initiale), puis en zero (la finale)."""
    produit = P * transformee()
    return sp.limit(produit, P, sp.oo), sp.limit(produit, P, 0)
