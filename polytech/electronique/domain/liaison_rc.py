"""La liaison capacitive : le passe-haut que `C` forme avec la charge `R`."""

from __future__ import annotations

import math


def capacite_de_liaison(resistance: float, frequence_de_coupure: float) -> float:
    """`C = 1 / (2 pi R f_c)` : la coupure du passe-haut `R C` tombe en `f_c`."""
    if resistance <= 0 or frequence_de_coupure <= 0:
        msg = f"non positive : R = {resistance}, f = {frequence_de_coupure}"
        raise ValueError(msg)
    return 1 / (2 * math.pi * resistance * frequence_de_coupure)
