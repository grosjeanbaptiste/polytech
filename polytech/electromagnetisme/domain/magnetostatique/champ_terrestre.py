"""Le champ magnetique terrestre.

Le syllabus d'electricite generale (10.17) : « dans nos regions, la
composante horizontale du champ (dirigee vers le pole nord magnetique)
est de l'ordre de 15 A/m et la composante verticale (dirigee vers le
centre de la terre) est de l'ordre de 35 A/m ».
"""

from __future__ import annotations

import math
from typing import Final

COMPOSANTE_HORIZONTALE: Final = 15.0
"""En amperes par metre, vers le pole nord magnetique."""

COMPOSANTE_VERTICALE: Final = 35.0
"""En amperes par metre, vers le centre de la terre."""


def intensite(horizontale: float, verticale: float) -> float:
    """La grandeur du champ, en amperes par metre."""
    return math.hypot(horizontale, verticale)


def inclinaison(horizontale: float, verticale: float) -> float:
    """L'angle du champ sous l'horizontale, en radians."""
    if horizontale == 0 and verticale == 0:
        msg = "champ nul : aucune direction, donc aucune inclinaison"
        raise ValueError(msg)
    return math.atan2(verticale, horizontale)
