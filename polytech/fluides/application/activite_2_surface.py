"""Activite 2, (H9) : la vitesse de la surface d'un reservoir que l'on remplit.

Le reservoir est alimente au debit `Q`. La masse donne
`A_T V_s = Q - A_o V_o`, soit `V_s = r^2 (V_eq - V_o)` avec `r = D_o / D_T`
et `V_eq = Q / A_o` la vitesse du jet a l'equilibre. Bernoulli complet,
de la surface au jet : `V_o^2 - V_s^2 = 2 g z`.

A `z_max`, `V_s = 0` : negliger la surface ne coute rien a la hauteur
maximale. Pendant la montee, le terme neglige rapporte a `V_o^2` vaut
`(V_s / V_o)^2 = r^4 (V_eq / V_o - 1)^2`, au plus `r^4` des que
`V_o >= V_eq / 2` -- ce qui est acquis pour `z >= z_max / 4`.
"""

from __future__ import annotations

import math


def _verifier_le_rapport(rapport: float) -> None:
    if not 0 < rapport < 1:
        msg = f"rapport des diametres hors de ]0, 1[ : {rapport}"
        raise ValueError(msg)


def borne_de_la_vitesse_de_surface(rapport: float) -> float:
    """`r^4` : la part au plus du terme neglige, des le quart de `z_max`."""
    _verifier_le_rapport(rapport)
    return rapport**4


def part_de_la_vitesse_de_surface(rapport: float, niveau_relatif: float) -> float:
    """`(V_s / V_o)^2` au niveau `z / z_max`, par la masse et Bernoulli.

    Avec `x = V_o / V_eq` et `s = z / z_max`, Bernoulli s'ecrit
    `x^2 - r^4 (1 - x)^2 = s` : un trinome en `x`, dont on garde la
    racine positive.
    """
    _verifier_le_rapport(rapport)
    if not 0 < niveau_relatif <= 1:
        msg = f"niveau relatif hors de ]0, 1] : {niveau_relatif}"
        raise ValueError(msg)
    r4 = rapport**4
    a = 1 - r4
    x = (-r4 + math.sqrt(r4**2 + a * (r4 + niveau_relatif))) / a
    return r4 * (1 / x - 1) ** 2
