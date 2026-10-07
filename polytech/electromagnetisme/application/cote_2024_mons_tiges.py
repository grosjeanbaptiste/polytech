"""Exercice cote 2024 (Mons), question 2 : une tige qui glisse sur deux tiges en V.

Deux tiges font l'angle `alpha` de part et d'autre de l'axe `x` ; une
troisieme, parallele a `y`, avance a la vitesse `v`. Le champ `(0, 0, b x)`
traverse le triangle qu'elles ferment.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.induction import (
    fem_de_deplacement,
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, ORDONNEE, TEMPS

T: Final = TEMPS
B, ALPHA, V, D, X = sp.symbols("b alpha v d x'", positive=True)
FEM_VOULUE: Final = 10.0
"""En volts."""
GRADIENT: Final = 0.1
"""`b`, en teslas par metre."""
DISTANCE: Final = 5.0
"""`d`, en metres."""
DEMI_ANGLE: Final = math.radians(20)


def flux() -> sp.Expr:
    """Le flux a travers le triangle, la troisieme tige etant en `x'`."""
    moitie = sp.integrate(
        B * ABSCISSE, (ORDONNEE, 0, ABSCISSE * sp.tan(ALPHA)), (ABSCISSE, 0, X)
    )
    return 2 * moitie


def fem_absolue() -> sp.Expr:
    """`|d(phi)/dt|`, avec `x' = v t`."""
    return sp.simplify(-force_electromotrice(flux().subs(X, V * T)))


def fem_a_la_distance() -> sp.Expr:
    """La meme, a l'instant `t = d/v` ou la tige passe en `x' = d`."""
    return sp.simplify(fem_absolue().subs(T, D / V))


def fem_par_la_tige() -> sp.Expr:
    """`B l v` : seule la tige mobile coupe des lignes de champ."""
    longueur = 2 * D * sp.tan(ALPHA)
    return B * D * longueur * V


def longueur_de_la_tige() -> float:
    """La portion de tige entre les deux autres, en metres, en `x' = d`."""
    return 2 * DISTANCE * math.tan(DEMI_ANGLE)


def vitesse() -> float:
    """La vitesse, en m/s, qui donne dix volts en `x' = d`."""
    champ = GRADIENT * DISTANCE
    return FEM_VOULUE / fem_de_deplacement(champ, longueur_de_la_tige(), 1.0)
