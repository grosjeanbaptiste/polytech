"""Le tenseur d'inertie : sa definition, sa rotation, le theoreme de Steiner.

Le tenseur d'inertie en `O` range les moments et les produits :

    I = [[ I_x, -P_xy, -P_xz], [-P_xy, I_y, -P_yz], [-P_xz, -P_yz, I_z]].

Quand les axes tournent, `1_XYZ = A 1_xyz`, il devient `A I A'`. Dans le
plan il existe un angle `theta*` qui annule le produit d'inertie :
`tan(2 theta*) = 2 P_xy / (I_y - I_x)` ; les axes ainsi tournes sont les
**axes principaux**. Quand le point change, le theoreme de Steiner donne
`I_O = I_G + m (a^2 delta - a_alpha a_beta)`, ou `a` repere `G` depuis `O` ;
pour un axe, `I_d = I_d1 + m d^2`.

Source : `MecaniqueRationnelle2/SYLLABUS/Syllabus_Theorie_2025_2026.pdf`,
sections 6.3 a 6.6, pages 109 a 114, relations (6.35), (6.44) a (6.49),
(6.58) et (6.60).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.mecanique.domain.inertie.moments import (
    moments_axiaux,
    produits_d_inertie,
)

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.produit_vectoriel import Vecteur
    from polytech.mecanique.domain.inertie.repartition import Repartition


def tenseur_d_inertie(solide: Repartition) -> sp.Matrix:
    """La matrice (6.35), au point origine et dans les axes du solide."""
    ix, iy, iz = moments_axiaux(solide)
    pxy, pyz, pxz = produits_d_inertie(solide)
    return sp.Matrix([[ix, -pxy, -pxz], [-pxy, iy, -pyz], [-pxz, -pyz, iz]])


def rotation_autour_de_z(angle: sp.Expr) -> sp.Matrix:
    """La matrice `A` de (6.45) : les axes tournent de `angle` autour de `z`."""
    s, c = sp.sin(angle), sp.cos(angle)
    return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def tourne(tenseur: sp.Matrix, rotation: sp.Matrix) -> sp.Matrix:
    """`I_XYZ = A I_xyz A'`, relation (6.44)."""
    return sp.simplify(rotation * tenseur * rotation.T)


def angle_principal(ix: sp.Expr, iy: sp.Expr, pxy: sp.Expr) -> sp.Expr:
    """`theta*`, relation (6.49) ; leve si toute direction est principale."""
    if ix == iy and pxy == 0:
        msg = "moments egaux et produit nul : toute direction est principale"
        raise ValueError(msg)
    return sp.atan2(2 * pxy, iy - ix) / 2


def steiner(au_centre: sp.Matrix, masse: sp.Expr, centre: Vecteur) -> sp.Matrix:
    """`I_O = I_G + m (a^2 delta - a_alpha a_beta)`, relation (6.58)."""
    a = sp.Matrix(centre)
    return sp.simplify(au_centre + masse * (a.dot(a) * sp.eye(3) - a * a.T))


def steiner_pour_un_axe(central: sp.Expr, masse: sp.Expr, distance: sp.Expr) -> sp.Expr:
    """`I_d = I_d1 + m d^2`, relation (6.60) : l'axe central porte le minimum."""
    return central + masse * distance**2
