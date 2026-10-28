"""Les profils de temperature le long d'un echangeur a courants paralleles.

L'ecart des deux fluides decroit en `exp(-N (1 + R) x)` (`x` reduit a `[0, 1]`,
`N = K S / C_1`, `R = C_1 / C_2`) ; le bilan partage la baisse du chaud et la
hausse du froid, dans le rapport `R`.
"""

from __future__ import annotations

import math


def profils_cocourants(
    chaude_entree: float,
    froide_entree: float,
    *,
    unites: float,
    rapport: float,
    position: float,
) -> tuple[float, float]:
    """`(t_chaud, t_froid)` a la position reduite `position`."""
    if not 0 <= position <= 1:
        msg = f"position hors de [0, 1] : {position}"
        raise ValueError(msg)
    if unites <= 0 or rapport < 0:
        msg = f"unites {unites} ou rapport {rapport} invalide"
        raise ValueError(msg)
    ecart_initial = chaude_entree - froide_entree
    perdu = (
        ecart_initial * -math.expm1(-unites * (1 + rapport) * position) / (1 + rapport)
    )
    return chaude_entree - perdu, froide_entree + rapport * perdu
