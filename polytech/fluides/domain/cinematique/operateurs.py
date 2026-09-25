"""Cours III.2, III.4 et III.5 -- operateurs differentiels sur un champ plan.

Chaque question du cours se teste par un operateur different, et ils sont
independants les uns des autres : c'est la raison d'etre de ce module.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y


def divergence(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """Taux de dilatation volumique relatif d'une particule (cours III.4)."""
    return sp.simplify(sp.diff(u, X) + sp.diff(v, Y))


def taux_de_rotation(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """Vitesse angulaire propre ``omega_z`` de la particule (cours III.5)."""
    return sp.simplify((sp.diff(v, X) - sp.diff(u, Y)) / 2)


def vorticite(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """Vorticite ``zeta_z = 2 omega_z`` (cours III.5)."""
    return sp.simplify(2 * taux_de_rotation(u, v))


def tenseur_taux_deformation(u: sp.Expr, v: sp.Expr) -> sp.Matrix:
    """Tenseur symetrique des taux de deformation (cours III.4)."""
    exx = sp.diff(u, X)
    eyy = sp.diff(v, Y)
    exy = (sp.diff(u, Y) + sp.diff(v, X)) / 2
    return sp.simplify(sp.Matrix([[exx, exy], [exy, eyy]]))


def derivee_materielle(
    grandeur: sp.Expr,
    u: sp.Expr,
    v: sp.Expr,
) -> sp.Expr:
    """Derivee particulaire ``D/Dt = d/dt + u d/dx + v d/dy`` (cours III.2).

    C'est l'operateur qui distingue stationnarite et acceleration nulle : le
    terme local peut etre nul sans que le terme convectif le soit.
    """
    return sp.simplify(
        sp.diff(grandeur, T) + u * sp.diff(grandeur, X) + v * sp.diff(grandeur, Y),
    )
