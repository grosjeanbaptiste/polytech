"""L'ecoulement en conduite : Poiseuille, et les pertes de charge regulieres.

Chapitre V du cours, « Ecoulements en conduite », que l'examen de janvier
2025 interroge et que le depot ne porte pas encore : ce module en tient ce
que l'examen demande.

* **Poiseuille** : dans une conduite circulaire de rayon `R`, en regime
  laminaire etabli, Navier-Stokes se reduit a `(mu / r) (r u')' = -G`, ou
  `G = -dp/dx` est le gradient de pression moteur. Le profil est
  parabolique, `u = G (R^2 - r^2) / (4 mu)`, le debit `Q = pi R^4 G / (8 mu)`,
  et la perte de charge sur une longueur `L`, `Delta p = 8 mu L Q / (pi R^4)`
  -- proportionnelle a la viscosite a debit donne.
* **Darcy-Weisbach** : `Delta p = lambda (L / D) rho V^2 / 2`, avec
  `lambda = 64 / Re` en laminaire et la loi de Blasius,
  `lambda = 0,316 Re^(-1/4)`, en turbulent lisse -- celle que l'enonce de
  2025 donne.

Une perte de charge est toujours positive : le frottement est irreversible
(planches des equations fondamentales, Bernoulli generalise).
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.fluides.domain.regime_de_reynolds import (
    REYNOLDS_LAMINAIRE,
    REYNOLDS_TURBULENT,
)

COEFFICIENT_DE_BLASIUS: Final = 0.316
"""`lambda = 0,316 Re^(-1/4)`, tel que l'enonce de 2025 l'ecrit."""


def profil_de_poiseuille(
    rayon_courant: sp.Expr, rayon: sp.Expr, viscosite: sp.Expr, gradient: sp.Expr
) -> sp.Expr:
    """`u(r) = G (R^2 - r^2) / (4 mu)`."""
    return gradient * (rayon**2 - rayon_courant**2) / (4 * viscosite)


def debit_de_poiseuille(
    rayon: sp.Expr, viscosite: sp.Expr, gradient: sp.Expr
) -> sp.Expr:
    """`Q = pi R^4 G / (8 mu)`."""
    return sp.pi * rayon**4 * gradient / (8 * viscosite)


def perte_de_charge_de_poiseuille(
    debit: float, rayon: float, longueur: float, viscosite: float
) -> float:
    """`Delta p = 8 mu L Q / (pi R^4)`, en pascals."""
    return 8 * viscosite * longueur * debit / (math.pi * rayon**4)


def coefficient_laminaire(reynolds: float) -> float:
    """`lambda = 64 / Re` ; leve hors du regime laminaire."""
    if not 0 < reynolds < REYNOLDS_LAMINAIRE:
        msg = f"64 / Re ne vaut qu'en laminaire : Re = {reynolds}"
        raise ValueError(msg)
    return 64 / reynolds


def coefficient_de_blasius(reynolds: float) -> float:
    """`lambda = 0,316 Re^(-1/4)` ; leve hors du regime turbulent."""
    if reynolds <= REYNOLDS_TURBULENT:
        msg = f"Blasius ne vaut qu'en turbulent : Re = {reynolds}"
        raise ValueError(msg)
    return float(COEFFICIENT_DE_BLASIUS * reynolds**-0.25)


def perte_de_charge_reguliere(
    coefficient: float,
    longueur: float,
    diametre: float,
    vitesse: float,
    masse_volumique: float,
) -> float:
    """Darcy-Weisbach : `lambda (L / D) rho V^2 / 2`, en pascals."""
    if longueur <= 0 or diametre <= 0:
        msg = f"longueur ou diametre non positif : L = {longueur}, D = {diametre}"
        raise ValueError(msg)
    return coefficient * longueur / diametre * masse_volumique * vitesse**2 / 2
