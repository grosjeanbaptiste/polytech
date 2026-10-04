"""La loi de refroidissement de Newton.

Un corps de temperature uniforme `T`, refroidi par convection dans un
milieu a `T_inf`, perd sa chaleur en proportion de l'ecart :

    dT/dt = -beta (T - T_inf),     T(t) = T_inf + (T_0 - T_inf) e^(-beta t).

Deux mesures separees d'un intervalle connu donnent `beta` ; la
temperature de depart donne alors le temps ecoule. C'est le corps
thermiquement mince du cours de thermique, avec `beta = 1 / tau`.
"""

from __future__ import annotations

import math


def _rapport_des_ecarts(avant: float, apres: float, ambiante: float) -> float:
    """`(T_avant - T_inf) / (T_apres - T_inf)`, plus grand que 1 par la loi."""
    ecart_avant, ecart_apres = avant - ambiante, apres - ambiante
    if ecart_avant * ecart_apres <= 0 or abs(ecart_apres) >= abs(ecart_avant):
        msg = (
            f"le corps ne se rapproche pas de l'ambiante : {avant} puis {apres} "
            f"pour {ambiante}"
        )
        raise ValueError(msg)
    return ecart_avant / ecart_apres


def taux_de_refroidissement(
    premiere: float, seconde: float, ambiante: float, intervalle: float
) -> float:
    """`beta = ln((T_1 - T_inf) / (T_2 - T_inf)) / intervalle`."""
    return math.log(_rapport_des_ecarts(premiere, seconde, ambiante)) / intervalle


def delai_de_refroidissement(
    initiale: float, mesuree: float, ambiante: float, taux: float
) -> float:
    """Le temps mis pour passer de `T_0` a la temperature mesuree."""
    return math.log(_rapport_des_ecarts(initiale, mesuree, ambiante)) / taux


def temperature_du_corps(
    initiale: float, ambiante: float, taux: float, duree: float
) -> float:
    """`T_inf + (T_0 - T_inf) e^(-beta t)`."""
    return ambiante + (initiale - ambiante) * math.exp(-taux * duree)
