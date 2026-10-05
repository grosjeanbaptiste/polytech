"""Le remplissage d'un reservoir perce au fond.

Exercice du chapitre IV.5 des planches. L'eau entre dans un reservoir de
section `A_T` au debit massique `m_point` ; un orifice a entree arrondie,
de section `A_o`, la laisse s'echapper au fond, sans perte, a la vitesse
de Torricelli `racine(2 g z)`. Le bilan de masse donne

`rho A_T dz/dt = m_point - rho A_o racine(2 g z)`.

Le niveau monte donc jusqu'a la hauteur ou l'orifice evacue tout le debit
entrant, et ne l'atteint qu'asymptotiquement. Parti de zero, il passe au
niveau `z` a l'instant

`t = -(2 / b) racine(z) - (2 a / b^2) ln(1 - b racine(z) / a)`,

avec `a = m_point / (rho A_T)` et `b = (A_o / A_T) racine(2 g)`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.fluides.domain.fondamentales.bernoulli import GRAVITE_EXACTE
from polytech.fluides.domain.grandeurs import PESANTEUR

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def hauteur_maximale_de_remplissage(
    debit_massique: float, masse_volumique: float, diametre_de_l_orifice: float
) -> float:
    """`z_max = (m_point / (rho A_o))^2 / (2 g)`, en metres."""
    if diametre_de_l_orifice <= 0 or masse_volumique <= 0:
        msg = (
            f"orifice ou masse volumique nuls ou negatifs : "
            f"{diametre_de_l_orifice} m, {masse_volumique} kg/m^3"
        )
        raise ValueError(msg)
    section = math.pi * diametre_de_l_orifice**2 / 4
    return (debit_massique / (masse_volumique * section)) ** 2 / (2 * PESANTEUR)


def duree_de_remplissage(
    niveau: Grandeur,
    debit_massique: Grandeur,
    masse_volumique: Grandeur,
    section: Grandeur,
    orifice: Grandeur,
) -> sp.Expr:
    """`t(z)` : le temps que met le niveau, parti de zero, a atteindre `z`."""
    apport = debit_massique / (masse_volumique * section)
    fuite = orifice / section * sp.sqrt(2 * GRAVITE_EXACTE)
    racine = sp.sqrt(niveau)
    logarithme = sp.log(1 - fuite * racine / apport)
    return -2 * racine / fuite - 2 * apport / fuite**2 * logarithme
