"""Ce que l'inertie fait osciller : le pendule compose, le ressort spiral.

Aux petites amplitudes le pendule compose obeit a
`I theta'' + m g r_G theta = 0`, d'ou sa periode
`T_c = 2 pi racine(I / (m g r_G))`. Elle est celle d'un pendule simple de
longueur `L = I / (m r_G)` : c'est la condition de Huygens. Un solide
rappele par un couple `-K theta` oscille avec `T = 2 pi racine(I / K)`,
ce qui permet de mesurer son moment d'inertie.

Source : `MecaniqueRationnelle2/SYLLABUS/Syllabus_Theorie_2025_2026.pdf`,
section 6.1.2, pages 104 et 105, relations (6.7) a (6.11) ; section 6.7.7,
page 116, relations (6.61) a (6.63).
"""

from __future__ import annotations

import sympy as sp

from polytech.mecanique.domain.pesanteur import PESANTEUR


def longueur_du_pendule_simple_isochrone(
    inertie: sp.Expr, masse: sp.Expr, distance_du_centre: sp.Expr
) -> sp.Expr:
    """`L = I / (m r_G)`, relation (6.7) ; leve si `G` est sur l'axe."""
    if masse * distance_du_centre == 0:
        msg = "centre de masse sur l'axe, ou masse nulle : aucun rappel"
        raise ValueError(msg)
    return inertie / (masse * distance_du_centre)


def periode_du_pendule_compose(
    inertie: sp.Expr,
    masse: sp.Expr,
    distance_du_centre: sp.Expr,
    pesanteur: sp.Expr | float = PESANTEUR,
) -> sp.Expr:
    """`T_c = 2 pi racine(I / (m g r_G))`, relation (6.10)."""
    longueur = longueur_du_pendule_simple_isochrone(inertie, masse, distance_du_centre)
    return 2 * sp.pi * sp.sqrt(longueur / pesanteur)


def periode_de_torsion(inertie: sp.Expr, raideur: sp.Expr) -> sp.Expr:
    """`T = 2 pi racine(I / K)`, relation (6.63)."""
    if raideur == 0:
        msg = "raideur de torsion nulle : le solide n'est pas rappele"
        raise ValueError(msg)
    return 2 * sp.pi * sp.sqrt(inertie / raideur)
