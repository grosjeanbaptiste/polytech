"""Le calcul des forces magnetiques, direct puis par l'energie.

Le syllabus d'electricite generale (chapitre 11). Directement, la force
volumique vaut `f = (i + i') ^ B` (11.1.3), les courants fictifs `i'`
portant l'aimantation.

Indirectement, deux theoremes. A flux constants, le travail des forces du
champ se fait au detriment de l'energie magnetique : `dT = -(dW_m)_phi`
(11.2.6). A courants constants, il se fait a son benefice :
`dT = +(dW_m)_I` (11.2.7), les sources fournissant a la fois le travail
et l'accroissement d'energie, `dW_s = -2 dT` (11.2.9). Les deux variations
sont opposees (11.2.5), ce que l'identite de Gauss etablit.

Les circuits tendent donc a se deplacer de maniere a embrasser le plus
grand flux possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

    Triplet = tuple[Any, Any, Any]


def force_volumique(courant: Triplet, fictif: Triplet, induction: Triplet) -> Triplet:
    """`f = (i + i') ^ B` (11.1.3), en newtons par metre cube."""
    x, y, z = (reel + lie for reel, lie in zip(courant, fictif, strict=True))
    b_x, b_y, b_z = induction
    return (y * b_z - z * b_y, z * b_x - x * b_z, x * b_y - y * b_x)


def _demi_somme(gauche: Sequence[Any], droite: Sequence[Any]) -> sp.Expr:
    paires = zip(gauche, droite, strict=True)
    return sp.Add(*(a * b for a, b in paires)) / 2


def variation_a_flux_constants(
    variations_des_courants: Sequence[Any], flux: Sequence[Any]
) -> sp.Expr:
    """`(dW_m)_phi = (1/2) somme des dI_k phi_k` (11.2.2)."""
    return _demi_somme(variations_des_courants, flux)


def variation_a_courants_constants(
    courants: Sequence[Any], variations_des_flux: Sequence[Any]
) -> sp.Expr:
    """`(dW_m)_I = (1/2) somme des I_k d(phi_k)` (11.2.3)."""
    return _demi_somme(courants, variations_des_flux)


def bilan_a_courants_constants(travail: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`((dW_m)_I, dW_s) = (dT, -2 dT)` : le gain du champ, la perte des sources."""
    return (travail, -2 * travail)


def couple_par_l_energie(
    energie: sp.Expr, angle: sp.Symbol, *, a_courants_constants: bool
) -> sp.Expr:
    """`C = +(dW_m/d(alpha))_I = -(dW_m/d(alpha))_phi` (11.2.13)."""
    signe = 1 if a_courants_constants else -1
    return sp.simplify(signe * sp.diff(energie, angle))
