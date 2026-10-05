"""Ce qu'un filtre doit tenir, et ce que sa carte des poles en dit.

Chapitre 3, partie I, planches 11 a 24 et 50 a 57.

* Le module se lit sur la carte : produit des distances de `j w` aux
  zeros, divise par le produit des distances aux poles.
* La perte d'insertion `L_A = 10 log10(|T|^-2)` est la forme que les
  tables et les logiciels emploient.
* Une fois l'ondulation fixee, l'ordre se cherche en l'incrementant.
* Le retard de groupe d'un filtre tout-poles est la somme des retards de
  ses poles ; constant, il translate le signal sans le deformer.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def module_par_les_distances(
    pulsation: float,
    *,
    poles: Sequence[complex],
    zeros: Sequence[complex] = (),
    gain: float = 1.0,
) -> float:
    """`|a_M| prod |j w - z_i| / prod |j w - p_i|`."""
    point = 1j * pulsation
    distances = math.prod(abs(point - pole) for pole in poles)
    if distances == 0:
        msg = f"pole sur l'axe imaginaire en {pulsation} rad/s : module infini"
        raise ValueError(msg)
    return abs(gain) * math.prod(abs(point - zero) for zero in zeros) / distances


def perte_d_insertion(module: float) -> float:
    """`L_A = 10 log10(|T|^-2)`, en decibels."""
    if module <= 0:
        msg = f"module nul ou negatif : {module}"
        raise ValueError(msg)
    return -20 * math.log10(module)


def ordre_minimal(
    module: Callable[[int], float], plafond: float, *, maximum: int = 50
) -> int:
    """Le premier `N` pour lequel `module(N) <= plafond`."""
    for ordre in range(1, maximum + 1):
        if module(ordre) <= plafond:
            return ordre
    msg = f"aucun ordre jusqu'a {maximum} ne tient le plafond {plafond}"
    raise ValueError(msg)


def retard_de_groupe(poles: Sequence[complex], pulsation: float) -> float:
    """`-d(arg T)/dw = somme -Re(p) / (Re(p)^2 + (w - Im(p))^2)`, tout-poles."""
    return sum(
        -pole.real / (pole.real**2 + (pulsation - pole.imag) ** 2) for pole in poles
    )


def signal_carre_filtre(
    instant: float,
    frequence: float,
    filtre: Callable[[float], complex],
    *,
    harmoniques: int,
) -> float:
    """`(4/pi) somme_(n impair) |H| sin(2 pi n f_o t + arg H) / n`, tronquee."""
    total = 0.0
    for rang in range(1, harmoniques + 1, 2):
        pulsation = 2 * math.pi * rang * frequence
        reponse = filtre(pulsation)
        phase = pulsation * instant + cmath.phase(reponse)
        total += abs(reponse) * math.sin(phase) / rang
    return 4 * total / math.pi
