"""Le nombre de Reynolds et les trois regimes de l'experience de 1883.

Chapitre I.3 des planches. Reynolds injecte un filet d'encre dans de
l'eau qui circule dans un tube. Le rapport des forces d'inertie aux
forces de viscosite, `Re = U H / nu`, decide de ce que devient le filet :

- sous 2 100, le regime est **laminaire** : les filaments restent
  distincts et alignes sur l'ecoulement, sur toute la longueur du tube ;
- au-dessus de 4 000, il est **turbulent** : l'encre se diffuse sur toute
  la section ;
- entre les deux, c'est la zone de **transition**.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

REYNOLDS_LAMINAIRE: Final = 2100.0
"""Sous ce nombre, l'ecoulement en conduite est laminaire."""

REYNOLDS_TURBULENT: Final = 4000.0
"""Au-dessus de ce nombre, il est turbulent."""


class RegimeDeReynolds(Enum):
    """Les trois regimes d'un ecoulement en conduite."""

    LAMINAIRE = "laminaire"
    TRANSITION = "transition"
    TURBULENT = "turbulent"


def nombre_de_reynolds(vitesse: float, longueur: float, viscosite: float) -> float:
    """`Re = U H / nu` : forces d'inertie sur forces de viscosite."""
    if viscosite <= 0:
        msg = f"viscosite cinematique nulle ou negative : {viscosite} m^2/s"
        raise ValueError(msg)
    return vitesse * longueur / viscosite


def regime_en_conduite(reynolds: float) -> RegimeDeReynolds:
    """Laminaire sous 2 100, turbulent au-dessus de 4 000, transition entre."""
    if reynolds < 0:
        msg = f"nombre de Reynolds negatif : {reynolds}"
        raise ValueError(msg)
    if reynolds < REYNOLDS_LAMINAIRE:
        return RegimeDeReynolds.LAMINAIRE
    if reynolds > REYNOLDS_TURBULENT:
        return RegimeDeReynolds.TURBULENT
    return RegimeDeReynolds.TRANSITION
