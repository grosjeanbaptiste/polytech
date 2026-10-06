"""La loi de Poisson, et sa limite normale.

`P(K = k) = lambda^k e^(-lambda) / k!` compte des evenements rares et
independants -- les electrons qui franchissent une barriere pendant une
duree donnee. Son esperance et sa variance valent toutes deux `lambda`.
Quand `lambda` est grand, un millier par exemple, elle se confond avec la
loi normale `N(lambda, lambda)`.
"""

from __future__ import annotations

import math


def probabilite_de_poisson(rang: int, parametre: float) -> float:
    """`lambda^k e^(-lambda) / k!`, calculee par son logarithme."""
    if rang < 0:
        msg = f"rang negatif : {rang}"
        raise ValueError(msg)
    if parametre <= 0:
        msg = f"parametre nul ou negatif : {parametre}"
        raise ValueError(msg)
    logarithme = rang * math.log(parametre) - parametre - math.lgamma(rang + 1)
    return math.exp(logarithme)
