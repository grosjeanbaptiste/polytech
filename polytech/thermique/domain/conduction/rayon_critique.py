"""Le rayon critique d'un isolant autour d'un tube.

Isoler ajoute une resistance de conduction `ln(r_e / r_1) / (2 pi lambda)`
mais agrandit la surface d'echange : la resistance de surface
`1 / (2 pi r_e K_e)` baisse. Leur somme est minimale (le flux maximal)
quand sa derivee en `r_e` s'annule, c'est-a-dire en `r_e = lambda / K_e`.
"""

from __future__ import annotations

from polytech.noyau import core


def rayon_critique(conductivite: float, echange: float) -> float:
    """`r_c = lambda / K_e`, en metres."""
    return core.rayon_critique(conductivite, echange)
