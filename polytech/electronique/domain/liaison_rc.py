"""La liaison capacitive : le passe-haut que `C` forme avec la charge `R`."""

from __future__ import annotations

from polytech.noyau import core


def capacite_de_liaison(resistance: float, frequence_de_coupure: float) -> float:
    """`C = 1 / (2 pi R f_c)` : la coupure du passe-haut `R C` tombe en `f_c`."""
    return core.capacite_de_liaison(resistance, frequence_de_coupure)
