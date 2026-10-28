"""Les valeurs de catalogue d'un amplificateur operationnel, avec leur source.

Ce sont des ordres de grandeur que le cours donne, non des resultats : la
fiche les cite, et un test relit la planche pour qu'elles ne derivent pas.
"""

from __future__ import annotations

from typing import Final

SOURCE: Final = "ElectroniqueInstrumentation/SLIDES/Slides_2_part1.pdf"
"""Planches « Finite input impedance » et « Finite voltage gain »."""
R_IN_MINIMALE: Final = 1e6
"""`R_in >= 1 MOhm` : la regle generale du cours (LM741 : 2 MOhm)."""
GAIN_TYPIQUE: Final = (1e4, 1e6)
"""`A_v` dans `[10^4, 10^6]`, avec de grandes tolerances de fabrication."""


def gain_est_typique(gain: float) -> bool:
    """Vrai si ce gain en boucle ouverte est dans la plage de catalogue."""
    basse, haute = GAIN_TYPIQUE
    return basse <= gain <= haute
