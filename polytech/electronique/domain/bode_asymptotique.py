"""Ce que vaut une asymptote de Bode, et ou elle cesse de valoir.

Planches du chapitre 1, partie III (planches 65 a 71). Un pole ou un
zero reel en `-b` apporte au module le facteur `|j w + b|`, que le trace
asymptotique remplace par `b` avant la cassure et par `w` apres. L'ecart
est maximal a la cassure, 3 dB, et tombe a 0,26 dB a un quart ou a
quatre fois la cassure : c'est la plage ou l'asymptote est juste.

Pour la phase, l'approximation en marche -- zero avant, un quart de tour
apres -- demande une decade de part et d'autre : il reste 5,7 degres a
un dixieme et a dix fois la cassure.

Une paire de poles conjugues, `(j w)^2 + 2 j xi w_0 w + w_0^2`, garde
les memes asymptotes qu'un pole double, mais porte en `w_0` un pic de
`20 log(1 / (2 xi))` decibels.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

DECADE = 20.0
QUART_DE_TOUR = 90.0


def _exiger_un_rapport(rapport: float) -> None:
    if rapport <= 0:
        msg = f"rapport nul ou negatif : {rapport}"
        raise ValueError(msg)


def erreur_de_l_asymptote(rapport: float) -> float:
    """L'ecart, en dB, entre `|j w + b|` et son asymptote, pour `w / b` donne."""
    _exiger_un_rapport(rapport)
    exact = DECADE * math.log10(math.hypot(1.0, rapport))
    return exact - DECADE * math.log10(max(1.0, rapport))


def erreur_de_phase_de_l_asymptote(rapport: float) -> float:
    """L'ecart, en degres, entre `arg(j w + b)` et sa marche, pour `w / b` donne."""
    _exiger_un_rapport(rapport)
    exacte = math.degrees(math.atan(rapport))
    return exacte if rapport <= 1 else QUART_DE_TOUR - exacte


def phase_asymptotique(
    pulsation: float, *, zeros: Sequence[float], poles: Sequence[float]
) -> float:
    """La phase en marches, en degres ; `zeros` et `poles` sont les cassures."""
    avance = sum(1 for cassure in zeros if pulsation > cassure)
    retard = sum(1 for cassure in poles if pulsation > cassure)
    return QUART_DE_TOUR * (avance - retard)


def pic_du_second_ordre(amortissement: float) -> float:
    """`20 log10(1 / (2 xi))`, en dB, a la pulsation propre."""
    if amortissement <= 0:
        msg = f"amortissement nul ou negatif : {amortissement}"
        raise ValueError(msg)
    return DECADE * math.log10(1 / (2 * amortissement))
