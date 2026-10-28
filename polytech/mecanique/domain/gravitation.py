"""La gravitation universelle : `F = G m1 m2 / r^2`.

La constante vit ici, dans la discipline qui l'enseigne ; l'electromagnetisme
la compare a la force de Coulomb (syllabus d'electricite generale, 2.20.1).
"""

from __future__ import annotations

from typing import Final

CONSTANTE_GRAVITATIONNELLE: Final = 6.6743e-11
"""`G`, en N m^2 / kg^2 (CODATA 2018)."""


def force_gravitationnelle(masse_1: float, masse_2: float, distance: float) -> float:
    """`F = G m1 m2 / r^2`, en newtons ; masses en kg, distance en m."""
    if masse_1 < 0 or masse_2 < 0:
        msg = f"masse negative : {masse_1}, {masse_2}"
        raise ValueError(msg)
    if distance <= 0:
        msg = f"distance non strictement positive : {distance}"
        raise ValueError(msg)
    return CONSTANTE_GRAVITATIONNELLE * masse_1 * masse_2 / distance**2
