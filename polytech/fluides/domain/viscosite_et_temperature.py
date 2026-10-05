"""La viscosite en fonction de la temperature.

Chapitre I.2.1 des planches. Les deux tendances sont opposees, et pour
deux raisons differentes.

Dans un **gaz**, la vitesse des molecules croit avec la temperature : les
collisions sont plus nombreuses et la viscosite augmente, selon
`mu = a T^(1/2) / (1 + b / T)`.

Dans un **liquide**, la dilatation ecarte les molecules et la viscosite
diminue, selon `mu = a 10^(b / (T - c))`. C'est pourquoi il faut refroidir
l'huile d'un palier pour qu'il garde sa charge utile.

Les temperatures sont absolues ; les constantes `a`, `b`, `c` sont
propres a chaque fluide.
"""

from __future__ import annotations

import math


def viscosite_d_un_gaz(temperature: float, a: float, b: float) -> float:
    """`mu = a T^(1/2) / (1 + b / T)`, en Pa.s."""
    if temperature <= 0:
        msg = f"temperature absolue nulle ou negative : {temperature} K"
        raise ValueError(msg)
    return a * math.sqrt(temperature) / (1 + b / temperature)


def viscosite_d_un_liquide(temperature: float, a: float, b: float, c: float) -> float:
    """`mu = a 10^(b / (T - c))`, en Pa.s."""
    if temperature <= c:
        msg = f"temperature au plus egale a c : {temperature} K pour c = {c} K"
        raise ValueError(msg)
    return a * 10 ** (b / (temperature - c))
