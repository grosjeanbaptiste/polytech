"""La matrice ABCD d'une cellule en T, et ce que sa mise en cascade donne.

Mettre deux cellules bout a bout ne multiplie pas leurs fonctions de
transfert : la seconde charge la premiere. C'est precisement ce que la
representation ABCD sait dire, en se contentant d'un produit de matrices --
et c'est pourquoi le corrige la choisit.
"""

from __future__ import annotations

import sympy as sp


def cellule_en_t(
    impedance_serie_amont: sp.Expr,
    impedance_serie_aval: sp.Expr,
    impedance_shunt: sp.Expr,
) -> sp.Matrix:
    """Rend la matrice ABCD d'une cellule en T.

    Une cellule RC simple est le cas ou l'impedance aval est nulle : la
    resistance en amont, le condensateur en shunt, rien apres.
    """
    if impedance_shunt == 0:
        msg = "impedance shunt nulle : la cellule court-circuite sa sortie"
        raise ValueError(msg)
    z1, z2, z3 = impedance_serie_amont, impedance_serie_aval, impedance_shunt
    return sp.Matrix(
        [
            [sp.simplify(1 + z1 / z3), sp.simplify(z1 + z2 + z1 * z2 / z3)],
            [sp.simplify(1 / z3), sp.simplify(1 + z2 / z3)],
        ],
    )


def matrice_z_en_l(
    impedance_serie: sp.Expr,
    impedance_shunt: sp.Expr,
) -> sp.Matrix:
    """Matrice impedance d'une cellule en L : une serie, puis un shunt.

    Port 2 ouvert, tout le courant d'entree traverse la serie puis le shunt,
    d'ou `Z_11 = Z_1 + Z_2` ; la tension de sortie est celle du shunt seul,
    d'ou `Z_12 = Z_21 = Z_22 = Z_2`.
    """
    return sp.Matrix(
        [
            [impedance_serie + impedance_shunt, impedance_shunt],
            [impedance_shunt, impedance_shunt],
        ],
    )


def parametres_g_depuis_z(matrice: sp.Matrix) -> sp.Matrix:
    """Convertit une matrice impedance en matrice hybride inverse `G`.

    La conversion est celle des tables du cours ; l'interet de la poser ici
    est qu'elle ne suppose rien du circuit, et vaut donc pour la cellule en
    L comme pour toute autre.
    """
    z11, z12 = matrice[0, 0], matrice[0, 1]
    z21, z22 = matrice[1, 0], matrice[1, 1]
    if z11 == 0:
        msg = "impedance d'entree nulle a vide : la matrice G n'existe pas"
        raise ValueError(msg)
    return sp.Matrix(
        [
            [sp.simplify(1 / z11), sp.simplify(-z12 / z11)],
            [sp.simplify(z21 / z11), sp.simplify(z22 - z21 * z12 / z11)],
        ],
    )


def parametres_g_en_l(
    impedance_serie: sp.Expr,
    impedance_shunt: sp.Expr,
) -> sp.Matrix:
    """La matrice `G` de la cellule en L, par sa matrice impedance."""
    return parametres_g_depuis_z(matrice_z_en_l(impedance_serie, impedance_shunt))
