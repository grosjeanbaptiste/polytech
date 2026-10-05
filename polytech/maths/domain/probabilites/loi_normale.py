"""La loi normale, a une et a plusieurs dimensions.

`N(mu, sigma^2)` a pour densite `exp(-(x - mu)^2 / (2 sigma^2)) /
(racine(2 pi) sigma)`. A `k` dimensions, de moyenne `mu` et de matrice de
covariance `S` inversible :

    f(x) = exp(-(x - mu)^T S^-1 (x - mu) / 2) / racine((2 pi)^k det S).

Deux composantes d'un vecteur gaussien centre, de meme variance `sigma^2`
et de correlation `rho`, se conditionnent simplement : sachant que l'une
vaut `a`, l'autre suit `N(rho a, (1 - rho^2) sigma^2)`. Si `rho` est
proche de un, la premiere predit la seconde ; s'il est nul, elle n'en dit
rien.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


def densite_normale(point: sp.Expr, moyenne: sp.Expr, variance: sp.Expr) -> sp.Expr:
    """La densite de `N(moyenne, variance)` en `point`."""
    exacte = sp.sympify(variance)
    if exacte.is_positive is False:
        msg = f"variance nulle ou negative : {variance}"
        raise ValueError(msg)
    ecart = (point - moyenne) ** 2 / (2 * exacte)
    return sp.exp(-ecart) / sp.sqrt(2 * sp.pi * exacte)


def densite_normale_multivariee(
    point: Sequence[sp.Expr], moyenne: Sequence[sp.Expr], covariance: sp.Matrix
) -> sp.Expr:
    """La densite du vecteur gaussien non degenere de covariance donnee."""
    determinant = sp.simplify(covariance.det())
    if determinant == 0:
        msg = "matrice de covariance degeneree : la loi n'a pas de densite"
        raise ValueError(msg)
    ecart = sp.Matrix(point) - sp.Matrix(moyenne)
    forme = (ecart.T * covariance.inv() * ecart)[0, 0]
    dimension = len(point)
    return sp.exp(-forme / 2) / sp.sqrt((2 * sp.pi) ** dimension * determinant)


def loi_conditionnelle(
    valeur: sp.Expr, correlation: sp.Expr, variance: sp.Expr
) -> tuple[sp.Expr, sp.Expr]:
    """`(rho a, (1 - rho^2) sigma^2)` : la moyenne et la variance sachant `a`."""
    exacte = sp.sympify(correlation)
    if (sp.Abs(exacte) > 1) == sp.true:
        msg = f"correlation hors de [-1, 1] : {correlation}"
        raise ValueError(msg)
    return exacte * valeur, (1 - exacte**2) * variance


def demi_largeur_de_l_intervalle(probabilite: float) -> float:
    """Le `z` tel que `N(0, 1)` tombe dans `[-z, z]` avec cette probabilite."""
    if not 0 < probabilite < 1:
        msg = f"probabilite hors de ]0, 1[ : {probabilite}"
        raise ValueError(msg)
    return NormalDist().inv_cdf((1 + probabilite) / 2)
