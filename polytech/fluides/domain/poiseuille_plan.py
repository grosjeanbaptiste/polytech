"""Poiseuille entre deux plaques fixes distantes de `h`.

Navier-Stokes se reduit a `mu u'' = -G`, `G = -dp/dx` ; l'adherence en
`y = 0` et `y = h` donne `u = G y (h - y) / (2 mu)`, et le debit par unite
de largeur `G h^3 / (12 mu)`. Meme demarche que la conduite circulaire
(`conduite.py`), en geometrie plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp


def profil_plan(
    ordonnee: sp.Expr, hauteur: sp.Expr, viscosite: sp.Expr, gradient: sp.Expr
) -> sp.Expr:
    """`u(y) = G y (h - y) / (2 mu)`."""
    return gradient * ordonnee * (hauteur - ordonnee) / (2 * viscosite)


def debit_plan(hauteur: sp.Expr, viscosite: sp.Expr, gradient: sp.Expr) -> sp.Expr:
    """`q = G h^3 / (12 mu)`, par unite de largeur."""
    return gradient * hauteur**3 / (12 * viscosite)
