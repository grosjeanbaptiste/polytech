"""Les equations de la physique mathematique, ecrites avec le laplacien.

Mathematiques appliquees, 9.2 (page 157). `operateurs.py` porte le
laplacien et le d'alembertien, donc les equations de Laplace et des
ondes ; voici les cinq autres. Chaque fonction rend le **membre** qui
doit s'annuler : une fonction est solution quand il vaut zero.

* chaleur, ou diffusion (loi de Fick) : `Delta u = u_t / k` ;
* Helmholtz : `Delta u + lambda u = 0` ;
* Poisson : `Delta u = f` ;
* Schrodinger : `Delta u + alpha (E - V) u = 0` ;
* Klein-Gordon : `[] u + lambda^2 u = 0`.

« c, k, lambda, alpha, E designent des constantes ; f et V sont des
fonctions connues. »
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.edp.operateurs import dalembertien, laplacien

if TYPE_CHECKING:
    from collections.abc import Sequence


def de_la_chaleur(
    u: sp.Expr, espace: Sequence[sp.Symbol], temps: sp.Symbol, diffusivite: object
) -> sp.Expr:
    """`Delta u - u_t / k`."""
    return laplacien(u, espace) - sp.diff(u, temps) / sp.sympify(diffusivite)


def de_helmholtz(u: sp.Expr, espace: Sequence[sp.Symbol], valeur: object) -> sp.Expr:
    """`Delta u + lambda u`."""
    return laplacien(u, espace) + sp.sympify(valeur) * u


def de_poisson(u: sp.Expr, espace: Sequence[sp.Symbol], source: object) -> sp.Expr:
    """`Delta u - f`."""
    return laplacien(u, espace) - sp.sympify(source)


def de_schrodinger(
    u: sp.Expr,
    espace: Sequence[sp.Symbol],
    alpha: object,
    energie: object,
    potentiel: object,
) -> sp.Expr:
    """`Delta u + alpha (E - V) u`."""
    ecart = sp.sympify(energie) - sp.sympify(potentiel)
    return laplacien(u, espace) + sp.sympify(alpha) * ecart * u


def de_klein_gordon(
    u: sp.Expr,
    espace: Sequence[sp.Symbol],
    temps: sp.Symbol,
    celerite: object,
    valeur: object,
) -> sp.Expr:
    """`[] u + lambda^2 u`."""
    return dalembertien(u, espace, temps, celerite) + sp.sympify(valeur) ** 2 * u
