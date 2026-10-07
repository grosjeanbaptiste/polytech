"""Exercice cote 2026 (Mons), question 2 : une petite spire au centre d'une grande.

La spire 1, de rayon `a`, est dans le plan `z = 0` ; son courant tourne en
sens horaire vu d'en haut. La spire 2, de surface `dS`, est au centre ; sa
normale `u_n` fait l'angle `alpha` avec `-u_z`.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
    integrale_de_biot_et_savart,
)
from polytech.electromagnetisme.domain.magnetostatique.boucle import (
    induction_sur_l_axe,
)
from polytech.electromagnetisme.domain.magnetostatique.couple_des_bobines import (
    energie_de_deux_bobines,
)

MU0: Final = PERMEABILITE
I, IP, A, DS, ALPHA = sp.symbols("I I' a dS alpha", positive=True)  # noqa: E741 -- le courant de l'enonce
Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]


def induction_au_centre() -> Vecteur:
    """`B(O)` : sur l'axe, et vers le bas puisque le courant tourne en sens horaire."""
    return sp.Integer(0), sp.Integer(0), -induction_sur_l_axe(I, A, 0)


def integrale_sur_le_cercle() -> tuple[float, float, float]:
    """`integrale de dl ^ R / R^3` au centre d'un cercle de rayon un, sens horaire."""
    angle = sp.Symbol("t", real=True)
    cercle = (sp.cos(-angle), sp.sin(-angle), sp.Integer(0))
    return integrale_de_biot_et_savart(cercle, angle, 0.0, 2 * math.pi, (0, 0, 0))


def normale() -> Vecteur:
    """`u_n`, dans le plan de profil : vers le bas, inclinee de `alpha`."""
    return sp.sin(ALPHA), sp.Integer(0), -sp.cos(ALPHA)


def mutuelle() -> sp.Expr:
    """`L_M = phi_M / I`, le flux de `B(O)` a travers `dS` selon `u_n`."""
    flux = sp.Matrix(normale()).dot(sp.Matrix(induction_au_centre())) * DS
    return sp.simplify(flux / I)


def couple() -> sp.Expr:
    """`C = dW/d(alpha)` a courants constants, soit `I I' dL_M/d(alpha)`."""
    propre_un, propre_deux = sp.symbols("L_1 L_2", positive=True)
    energie = energie_de_deux_bobines(propre_un, propre_deux, mutuelle(), I, IP)
    return sp.simplify(energie.diff(ALPHA))


def couple_sur_le_moment() -> Vecteur:
    """`m x B`, avec le moment magnetique `m = I' dS u_n` de la spire 2."""
    moment = IP * DS * sp.Matrix(normale())
    x, y, z = moment.cross(sp.Matrix(induction_au_centre()))
    return sp.simplify(x), sp.simplify(y), sp.simplify(z)


def pente_de_l_angle_sous_le_couple() -> sp.Expr:
    """`d(alpha)/d(delta)` quand `u_n` tourne de `delta` autour de l'axe du couple.

    La rotation deplace `u_n` de `axe x u_n` ; on en tire la variation de
    `cos(alpha) = u_n . (-u_z)`, puis celle de `alpha`, pour un angle
    entre zero et un demi-tour, ou son sinus est positif.
    """
    axe = sp.Matrix(couple_sur_le_moment()).normalized()
    deplacement = axe.cross(sp.Matrix(normale()))
    variation_du_cosinus = deplacement.dot(sp.Matrix([0, 0, -1]))
    pente = sp.simplify(-variation_du_cosinus / sp.sin(ALPHA))
    return sp.refine(pente, sp.Q.positive(sp.sin(ALPHA)))
