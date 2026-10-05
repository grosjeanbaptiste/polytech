"""Les unites de la thermique, et celles qu'elle a quittees.

Syllabus de thermique, tome 1, section 0.7. Le seul systeme legal est le
Systeme international ; beaucoup d'ouvrages emploient encore le « systeme
pratique des thermiciens », metre, kilogramme, heure, kilocalorie.

`1 kcal = 4185,5 J`. Un flux en kilocalories par heure se convertit donc
par `4185,5 / 3600`, et le meme facteur vaut pour une densite de flux en
`kcal / (h m^2)` et pour un coefficient en `kcal / (h m^2 degre)`.
"""

from __future__ import annotations

from typing import Final

KILOCALORIE: Final = 4185.5
"""En joules."""

HEURE: Final = 3600.0
"""En secondes."""


def joules_depuis_kilocalories(quantite: float) -> float:
    """Une quantite de chaleur, de kilocalories en joules."""
    return quantite * KILOCALORIE


def watts_depuis_kilocalories_par_heure(flux: float) -> float:
    """Un flux, de kilocalories par heure en watts."""
    return flux * KILOCALORIE / HEURE
