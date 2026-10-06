"""Exercice cote 2022 (Charleroi), question 3 : basculer une bobine sur une resistance.

En position 1 la bobine est en serie avec la source et `R0`, en regime
sinusoidal ; en position 2 elle se decharge dans `R`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.phaseurs import valeur_instantanee
from polytech.electromagnetisme.domain.symboles import TEMPS

T = TEMPS
E, R, R0, L, W = sp.symbols("E R R_0 L omega", positive=True)
THETA = sp.Symbol("theta", real=True)


def courant_avant() -> sp.Expr:
    """Le courant de la bobine en position 1 : `I = E / (R0 + j omega L)`."""
    phaseur = E * sp.exp(sp.I * THETA) / (R0 + sp.I * W * L)
    return valeur_instantanee(phaseur, W)


def courant_apres() -> sp.Expr:
    """`i(t) = i(0) e^(-R t / L)` : le courant de la bobine ne saute pas."""
    depart = sp.simplify(courant_avant().subs(T, 0))
    return depart * sp.exp(-R * T / L)


def tension_de_la_bobine() -> sp.Expr:
    """`u_L(t)` pour `t > 0` : l'oppose de la tension de `R`, seule dans la boucle."""
    return sp.simplify(-R * courant_apres())


def tensions_au_basculement() -> tuple[sp.Expr, sp.Expr]:
    """`u_L` juste avant, puis juste apres : elle saute, le courant non."""
    avant = L * courant_avant().diff(T)
    return sp.simplify(avant.subs(T, 0)), sp.simplify(tension_de_la_bobine().subs(T, 0))
