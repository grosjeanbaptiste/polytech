"""Exercice cote 2018, question 4 : une commutation, et son systeme en Laplace.

Avant `t = 0` la source alimente `R` et `C` en serie, en regime
sinusoidal ; apres, `C` se decharge dans `R` et `L`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.phaseurs import valeur_instantanee
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    valeur_initiale,
)

T: Final = TEMPS
P: Final = VARIABLE
E, R, L, C, W = sp.symbols("E R L C omega", positive=True)
THETA = sp.Symbol("theta", real=True)
IP, UC = sp.symbols("I U_C")
"""Les images `I(p)` et `U_C(p)`."""
I0, U0 = sp.symbols("i_0 u_0", real=True)
"""`i(0)` et `u_C(0)`, laisses en lettres dans le systeme."""


def tension_avant() -> sp.Expr:
    """`u_C(t)` pour `t < 0` : le diviseur `1/(1 + j omega R C)` sur la source."""
    phaseur = E * sp.exp(sp.I * THETA) / (1 + sp.I * W * R * C)
    return valeur_instantanee(phaseur, W)


def tension_initiale() -> sp.Expr:
    """`u_C(0)`, la tension que le condensateur garde a la commutation."""
    return tension_avant().subs(T, 0)


def systeme() -> tuple[sp.Expr, sp.Expr]:
    """Les deux equations transformees, chacune egalee a zero."""
    maille = R * IP + L * (P * IP - I0) - UC
    condensateur = IP + C * (P * UC - U0)
    return sp.expand(maille), sp.expand(condensateur)


def image_du_courant() -> sp.Expr:
    """`I(p)`, la bobine partant sans courant : `i(0) = 0`."""
    resolues = sp.solve(systeme(), (IP, UC))
    return sp.simplify(resolues[IP].subs(I0, 0))


def depart_du_courant() -> tuple[sp.Expr, sp.Expr]:
    """`i(0+)` et `di/dt(0+)`, par le theoreme de la valeur initiale."""
    image = image_du_courant()
    depart = sp.simplify(valeur_initiale(image))
    return depart, sp.simplify(valeur_initiale(P * image - depart))
