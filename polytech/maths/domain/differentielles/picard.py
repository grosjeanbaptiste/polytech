"""Le probleme de Cauchy vu comme un point fixe : pourquoi Picard converge.

`y' = f(t, y)`, `y(t_0) = y_0` equivaut a `y = T(y)` avec

    T(y)(t) = y_0 + int_{t_0}^{t} f(tau, y(tau)) dtau.

Si `f` est **lipschitzienne** en `y` -- `|f(t, y) - f(t, z)| <= alpha |y - z|`
-- alors `||T y - T z|| <= alpha |t - t_0| d(y, z)` : `T` est contractante
pres de `t_0`, et le theoreme du point fixe donne l'existence, l'unicite et
une methode de calcul. Il ne garantit la solution que la ou `T` contracte :
`|t - t_0| <= h/M` (les iterees restent dans le domaine) et `<= K/alpha`.

Quand `f = a(t) y`, la constante se calcule au plus juste :
`K(delta) = int |a|` sur `[t_0, t_0 + delta]`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE
from polytech.maths.domain.series.uniforme import distance_uniforme

DEMI_LARGEUR: Final = sp.Symbol("delta", positive=True)
"""`delta`, la distance a `t_0`."""


def constante_de_lipschitz(
    second_membre: sp.Expr, domaine_x: sp.Set, domaine_y: sp.Set
) -> sp.Expr:
    """`sup |df/dy|` sur le domaine."""
    derivee = sp.diff(second_membre, INCONNUE)
    if derivee.has(VARIABLE) and derivee.has(INCONNUE):
        msg = f"df/dy = {derivee} depend a la fois de x et de y"
        raise ValueError(msg)
    if derivee.has(INCONNUE):
        return distance_uniforme(derivee, 0, INCONNUE, domaine_y)
    if derivee.has(VARIABLE):
        return distance_uniforme(derivee, 0, VARIABLE, domaine_x)
    return sp.Abs(derivee)


def contraction_de_picard(second_membre: sp.Expr, x0: object) -> sp.Expr:
    """`K(delta) = int_0^delta |a(x_0 + s)| ds`, pour `f = a(x) y`."""
    coefficient = sp.diff(second_membre, INCONNUE)
    if coefficient.has(INCONNUE):
        msg = f"second membre non lineaire en y : {second_membre}"
        raise ValueError(msg)
    pas = sp.Symbol("s", positive=True)
    decale = sp.Abs(coefficient.subs(VARIABLE, sp.sympify(x0) + pas))
    return sp.integrate(decale, (pas, 0, DEMI_LARGEUR))


def demi_largeur_garantie(second_membre: sp.Expr, x0: object) -> sp.Expr:
    """Le `delta` en deca duquel `K(delta) < 1`."""
    constante = contraction_de_picard(second_membre, x0)
    (solution,) = sp.solve(sp.Eq(constante, 1), DEMI_LARGEUR)
    return solution


def voisinage_de_cauchy(
    *, demi_cote: object, majorant: object, lipschitz: object, contraction: object
) -> tuple[sp.Expr, sp.Expr]:
    """`(h_1, h_2) = (h / M, K / alpha)` du theoreme de Cauchy."""
    k = sp.sympify(contraction)
    if not 0 <= k < 1:
        msg = f"K doit verifier 0 <= K < 1, recu {k}"
        raise ValueError(msg)
    return sp.sympify(demi_cote) / majorant, k / sp.sympify(lipschitz)
