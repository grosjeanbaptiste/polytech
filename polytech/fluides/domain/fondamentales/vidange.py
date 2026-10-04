"""La vidange d'un reservoir par un orifice : le « seau troue ».

La vitesse de sortie est celle de Torricelli, `V = racine(2 g h)`. Le
debit qui sort par l'orifice de section `s` fait baisser la surface
libre, de section `S` : `S h' = -s V`, soit

    h'(t) = gamma racine(h(t)),     gamma = -(s / S) racine(2 g) < 0.

En partant de `h_0`, la hauteur decroit comme un carre,
`h(t) = gamma^2 / 4 (t - alpha)^2`, jusqu'a l'instant `alpha =
2 racine(h_0) / |gamma|` ou le reservoir est vide ; elle reste nulle
ensuite.

Le cours de mathematique s'en sert a rebours : un seau vide a l'instant
`t_1` a pu se vider a n'importe quel instant `alpha <= t_1` -- le
probleme a une infinite de solutions.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.fondamentales.bernoulli import GRAVITE_EXACTE


def coefficient_de_vidange(orifice: object, surface: object) -> sp.Expr:
    """`gamma = -(s / S) racine(2 g)`."""
    rapport = sp.sympify(orifice) / sp.sympify(surface)
    return -rapport * sp.sqrt(2 * GRAVITE_EXACTE)


def duree_de_vidange(hauteur: object, orifice: object, surface: object) -> sp.Expr:
    """`alpha = 2 racine(h_0) / |gamma|`."""
    gamma = coefficient_de_vidange(orifice, surface)
    return sp.simplify(2 * sp.sqrt(sp.sympify(hauteur)) / -gamma)


def hauteur_pendant_la_vidange(
    hauteur: object,
    orifice: object,
    surface: object,
    instant: object,
) -> sp.Expr:
    """`gamma^2 / 4 (t - alpha)^2` jusqu'a `alpha`, puis zero."""
    gamma = coefficient_de_vidange(orifice, surface)
    alpha = duree_de_vidange(hauteur, orifice, surface)
    t = sp.sympify(instant)
    return sp.Piecewise((gamma**2 / 4 * (t - alpha) ** 2, t <= alpha), (0, True))
