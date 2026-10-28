"""Un facteur spectral en marches, moyenne par la part d'energie de chaque bande.

`alpha = somme alpha_i [k(lambda_i T) - k(lambda_(i-1) T)]` : la temperature
est celle de la source pour l'absorptivite, celle du corps pour l'emissivite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.noyau import core
from polytech.thermique.domain.rayonnement.corps_noir import fraction_emise_sous

if TYPE_CHECKING:
    from collections.abc import Sequence


def absorptivite_par_bandes(
    facteurs: Sequence[float], coupures: Sequence[float], temperature: float
) -> float:
    """Moyenne de `facteurs` (un par bande) ; `coupures` en metres, croissantes.

    Il y a une bande de plus que de coupures : la derniere va a l'infini.
    """
    core.valider_bandes(list(facteurs), list(coupures))
    parts = [fraction_emise_sous(c, temperature) for c in coupures]
    return core.moyenne_par_bandes(list(facteurs), parts)
