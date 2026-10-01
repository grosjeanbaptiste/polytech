"""Le corps noir : Stefan-Boltzmann, Wien, et la loi de Planck.

Le formulaire donne ces trois-la sous forme de courbes -- la loi de Planck
tracee pour 300 K et 6000 K, et la loi integree en abaque de fraction. Les
expressions analytiques sont donc etablies ici, et chacune se verifie
contre les deux autres : l'integrale de Planck sur tout le spectre doit
redonner Stefan-Boltzmann, et son maximum doit tomber sur Wien. Ce sont
ces deux recoupements que les tests exigent.
"""

from __future__ import annotations

import math
from typing import Final

STEFAN_BOLTZMANN: Final = 5.670374419e-8
"""Constante de Stefan-Boltzmann, en W/(m^2 K^4)."""

WIEN: Final = 2.897771955e-3
"""Constante de deplacement de Wien, en m K."""

PLANCK: Final = 6.62607015e-34
"""Constante de Planck, en J s."""

BOLTZMANN: Final = 1.380649e-23
"""Constante de Boltzmann, en J/K."""

LUMIERE: Final = 2.99792458e8
"""Vitesse de la lumiere dans le vide, en m/s."""


def emittance(temperature: float) -> float:
    """`sigma T^4`, l'emittance totale du corps noir, en W/m^2."""
    if temperature < 0:
        msg = f"temperature absolue negative interdite : {temperature} K"
        raise ValueError(msg)
    return STEFAN_BOLTZMANN * temperature**4


def longueur_d_onde_du_maximum(temperature: float) -> float:
    """Loi de Wien : `lambda_max = 2898 micrometres K / T`, en metres."""
    if temperature <= 0:
        msg = f"temperature nulle ou negative : {temperature} K"
        raise ValueError(msg)
    return WIEN / temperature


def luminance_spectrale(longueur_d_onde: float, temperature: float) -> float:
    """Loi de Planck, en W/(m^2 m) -- emittance par unite de longueur d'onde."""
    if longueur_d_onde <= 0:
        msg = f"longueur d'onde nulle ou negative : {longueur_d_onde} m"
        raise ValueError(msg)
    numerateur = 2.0 * math.pi * PLANCK * LUMIERE**2
    exposant = PLANCK * LUMIERE / (longueur_d_onde * BOLTZMANN * temperature)
    return numerateur / (longueur_d_onde**5 * (math.expm1(exposant)))
