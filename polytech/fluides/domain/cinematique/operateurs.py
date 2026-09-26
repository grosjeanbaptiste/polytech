"""Cours III.2, III.4 et III.5 -- operateurs differentiels sur un champ plan.

Chaque question du cours se teste par un operateur different, et ils sont
independants les uns des autres : c'est la raison d'etre de ce module. Les
trois premiers ne sont pas des formules autonomes mais trois lectures du
meme gradient de vitesse (module ``decomposition``) -- sa trace, sa part
symetrique, sa part antisymetrique.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.decomposition import (
    gradient_de_vitesse,
    partie_antisymetrique,
    partie_symetrique,
)
from polytech.fluides.domain.cinematique.symboles import T, X, Y


def divergence(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """Trace du gradient : taux de dilatation volumique relatif (III.4)."""
    return sp.simplify(gradient_de_vitesse(u, v).trace())


def taux_de_rotation(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """Unique coefficient de la part antisymetrique, ``omega_z`` (III.5)."""
    antisymetrique = partie_antisymetrique(gradient_de_vitesse(u, v))
    resultat: sp.Expr = antisymetrique[1, 0]
    return resultat


def vorticite(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """Vorticite ``zeta_z = 2 omega_z`` (cours III.5)."""
    return sp.simplify(2 * taux_de_rotation(u, v))


def tenseur_taux_deformation(u: sp.Expr, v: sp.Expr) -> sp.Matrix:
    """Part symetrique du gradient : les taux de deformation (III.4)."""
    return partie_symetrique(gradient_de_vitesse(u, v))


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


def derivee_le_long_de_la_trajectoire(
    grandeur: sp.Expr,
    u: sp.Expr,
    v: sp.Expr,
) -> sp.Expr:
    """Derive ``grandeur`` en suivant la particule, par derivation en chaine.

    On remplace ``x`` et ``y`` par la position ``x(t), y(t)`` d'une particule,
    on derive en ``t``, puis on reconnait ``dx/dt = u`` et ``dy/dt = v``. Le
    resultat coincide avec ``derivee_materielle`` : c'est la *demonstration*
    de l'operateur, et non une seconde definition.
    """
    abscisse, ordonnee = sp.Function("x")(T), sp.Function("y")(T)
    mouvement = {X: abscisse, Y: ordonnee}
    vitesses = {
        sp.Derivative(abscisse, T): u.subs(mouvement),
        sp.Derivative(ordonnee, T): v.subs(mouvement),
    }
    derivee = sp.diff(grandeur.subs(mouvement), T).subs(vitesses)
    return sp.simplify(derivee.subs({abscisse: X, ordonnee: Y}))
