"""La zone completement rugueuse : `lambda` ne depend plus du Reynolds.

Planches 21 a 23 (diagramme de Moody). Quand l'aspérité sort de la
sous-couche visqueuse, la paroi seule fixe le frottement : dans
Colebrook, le terme `2,51 / (Re sqrt(lambda))` devient negligeable devant
`eps / (3,71 D)` et il reste
`1 / sqrt(lambda) = -2 log10(eps / (3,71 D))` -- la limite que
Nikuradse ecrit `1,14 - 0,868 ln(eps / D)`. Le critere de Nikuradse est
`Re_eps = Re (eps / D) sqrt(lambda / 8) > 70`.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.fluides.domain.pertes_de_charge import coefficient_de_colebrook

REYNOLDS_DE_RUGOSITE_RUGUEUX: Final = 70
"""Au-dela, la paroi est completement rugueuse (Nikuradse)."""


def coefficient_en_zone_rugueuse(rugosite_relative: float) -> float:
    """`lambda = (-2 log10(eps / (3,71 D)))^-2`, independant de `Re`."""
    if not 0 < rugosite_relative < 1:
        msg = f"rugosite relative hors de ]0 ; 1[ : eps / D = {rugosite_relative}"
        raise ValueError(msg)
    return float((-2 * math.log10(rugosite_relative / 3.71)) ** -2)


def reynolds_de_rugosite(
    reynolds: float, rugosite_relative: float, coefficient: float
) -> float:
    """`Re_eps = Re (eps / D) sqrt(lambda / 8)`."""
    if reynolds <= 0 or coefficient <= 0:
        msg = (
            f"Reynolds ou lambda non positif : Re = {reynolds}, lambda = {coefficient}"
        )
        raise ValueError(msg)
    return reynolds * rugosite_relative * math.sqrt(coefficient / 8)


def est_completement_rugueux(reynolds: float, rugosite_relative: float) -> bool:
    """Le `lambda` de Colebrook a ce `Re` franchit le seuil de 70."""
    coefficient = coefficient_de_colebrook(reynolds, rugosite_relative)
    return (
        reynolds_de_rugosite(reynolds, rugosite_relative, coefficient)
        > REYNOLDS_DE_RUGOSITE_RUGUEUX
    )
