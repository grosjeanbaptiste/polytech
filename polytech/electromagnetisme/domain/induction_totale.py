"""La force electromotrice induite totale, et la loi de Lenz.

Le syllabus d'electricite generale (14.3 et 14.4). La loi de Faraday,
`E_12 = -d(phi_12)/dt` (14.3.1), vaut quelle que soit la cause de la
variation du flux. La derivee se scinde en deux (14.3.2) : la force
electromotrice induite **statique** `-(d(phi)/dt)` a circuit immobile
(14.3.3), que pousse le champ `E_i = -dA/dt`, et la **dynamique**
(14.3.4), due au seul mouvement du circuit, que pousse `E_d = v ^ B`. La
distinction depend de l'observateur ; leur somme, non.

Le flux est ici une expression du temps et d'une coordonnee qui repere le
circuit.

Loi de Lenz : la force electromotrice a le signe oppose a celui de la
**variation** du flux, non du flux. C'est une consequence de la
conservation de l'energie.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS


def fem_induite_statique(flux: sp.Expr) -> sp.Expr:
    """`(E_12)_i = -d(phi)/dt` a circuit immobile (14.3.3)."""
    return -sp.diff(flux, TEMPS)


def fem_induite_dynamique(flux: sp.Expr, position: sp.Symbol, vitesse: Any) -> sp.Expr:  # noqa: ANN401 -- reel ou symbole
    """`(E_12)_d = -(d(phi)/dx) v` (14.3.4) : la part du mouvement du circuit."""
    return -sp.diff(flux, position) * vitesse


def sens_de_la_fem_induite(variation_du_flux: float) -> int:
    """Le signe de la force electromotrice : oppose a celui de la variation."""
    return int(-sp.sign(variation_du_flux))
