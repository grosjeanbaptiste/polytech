"""Exercice cote 2022 (Mons), question 1 : une spire tombe dans un champ non uniforme.

`B = (-B x, B x, B z)`. La spire, horizontale, part de la hauteur `h`
sans vitesse : seule la composante `B_z`, qui croit avec l'altitude,
traverse son plan.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, COTE, TEMPS

T: Final = TEMPS
B, R, H, G = sp.symbols("B R h g", positive=True)


def champ() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """L'induction de l'enonce, en coordonnees cartesiennes."""
    return -B * ABSCISSE, B * ABSCISSE, B * COTE


def divergence() -> sp.Expr:
    """`div B`, qui doit etre nulle pour que le champ soit magnetique."""
    return divergence_cartesienne(champ())


def altitude() -> sp.Expr:
    """`z(t) = h - g t^2/2` : une chute libre sans vitesse initiale."""
    return H - G * T**2 / 2


def flux() -> sp.Expr:
    """`phi = B_z pi R^2`, la normale de la spire etant `u_z`."""
    return champ()[2].subs(COTE, altitude()) * sp.pi * R**2


def flux_par_integration() -> sp.Expr:
    """Le meme flux, `B_z` integre sur le disque en coordonnees polaires."""
    rayon, angle = sp.symbols("rho phi", nonnegative=True)
    selon_z = champ()[2].subs(COTE, altitude())
    return sp.integrate(selon_z * rayon, (rayon, 0, R), (angle, 0, 2 * sp.pi))


def fem_absolue() -> sp.Expr:
    """`|E| = |-d(phi)/dt|` pendant la chute, donc pour `t >= 0`."""
    absolue = sp.Abs(force_electromotrice(flux()))
    return sp.refine(absolue, sp.Q.nonnegative(T))


def duree_de_la_chute() -> sp.Expr:
    """L'instant ou la spire touche le sol : `z = 0`."""
    return sp.sqrt(2 * H / G)
