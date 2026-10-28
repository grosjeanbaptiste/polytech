"""Un facteur spectral en marches, moyenne par la part d'energie de chaque bande.

`alpha = somme alpha_i [k(lambda_i T) - k(lambda_(i-1) T)]` : la temperature
est celle de la source pour l'absorptivite, celle du corps pour l'emissivite.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from polytech.thermique.domain.rayonnement.corps_noir import fraction_emise_sous

if TYPE_CHECKING:
    from collections.abc import Sequence


def absorptivite_par_bandes(
    facteurs: Sequence[float], coupures: Sequence[float], temperature: float
) -> float:
    """Moyenne de `facteurs` (un par bande) ; `coupures` en metres, croissantes.

    Il y a une bande de plus que de coupures : la derniere va a l'infini.
    """
    if len(facteurs) != len(coupures) + 1:
        msg = f"{len(facteurs)} facteurs, {len(coupures)} coupures (un de plus attendu)"
        raise ValueError(msg)
    if any(not 0 <= f <= 1 for f in facteurs):
        msg = f"facteur hors de [0, 1] : {list(facteurs)}"
        raise ValueError(msg)
    if any(b <= a for a, b in itertools.pairwise(coupures)):
        msg = f"coupures non croissantes : {list(coupures)}"
        raise ValueError(msg)
    parts = [fraction_emise_sous(c, temperature) for c in coupures]
    bornes = [0.0, *parts, 1.0]
    return sum(f * (bornes[i + 1] - bornes[i]) for i, f in enumerate(facteurs))
