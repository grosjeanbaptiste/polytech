"""Cours III.4 -- d'ou viennent le tenseur des taux de deformation et omega_z.

Le mouvement local d'une particule est entierement contenu dans le *gradient
de vitesse*. Toute matrice se decompose de facon unique en une part
symetrique et une part antisymetrique : la premiere est le tenseur des taux
de deformation, la seconde ne contient qu'un coefficient, le taux de
rotation. Les deux grandeurs que le cours enonce separement sont donc les
deux moities d'un meme objet, ce qui explique pourquoi elles sont
independantes l'une de l'autre.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import X, Y


def gradient_de_vitesse(u: sp.Expr, v: sp.Expr) -> sp.Matrix:
    """Matrice des quatre derivees spatiales du champ."""
    return sp.Matrix(
        [
            [sp.diff(u, X), sp.diff(u, Y)],
            [sp.diff(v, X), sp.diff(v, Y)],
        ],
    )


def partie_symetrique(matrice: sp.Matrix) -> sp.Matrix:
    """Part qui deforme la particule : ``(M + M^T) / 2``."""
    return sp.simplify((matrice + matrice.T) / 2)


def partie_antisymetrique(matrice: sp.Matrix) -> sp.Matrix:
    """Part qui la fait tourner en bloc : ``(M - M^T) / 2``."""
    return sp.simplify((matrice - matrice.T) / 2)
