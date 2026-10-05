"""La trainee en pratique : voitures, sphere, chute freinee.

Planches du chapitre IX. Le coefficient de trainee d'un corps se mesure,
et les planches en donnent quelques-uns. Pour une sphere en ecoulement
rampant il se calcule : `C_D = 24 / Re`, la loi de Stokes, qui ne tient
que tant que l'inertie est negligeable.

A coefficient constant la trainee croit comme le carre de la vitesse.
Un corps qui tombe accelere donc jusqu'a ce qu'elle equilibre son poids,
`m g = C_D A rho V^2 / 2`.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.fluides.domain.grandeurs import PESANTEUR

COEFFICIENTS_DE_TRAINEE_AUTOMOBILE: Final = {
    "Peugeot 5008": 0.28,
    "Porsche Cayenne": 0.36,
    "Audi A3": 0.31,
    "Tesla Model S": 0.24,
}
"""Le tableau de la planche 30."""

LIMITE_DE_STOKES: Final = 1.0
"""Au-dela de `Re = 1`, la courbe de la sphere quitte la droite `24 / Re`."""


def coefficient_de_stokes(reynolds: float) -> float:
    """`C_D = 24 / Re` pour une sphere, en ecoulement rampant."""
    if reynolds <= 0:
        msg = f"Reynolds nul ou negatif : {reynolds}"
        raise ValueError(msg)
    if reynolds > LIMITE_DE_STOKES:
        msg = f"hors du regime de Stokes : Re = {reynolds} depasse {LIMITE_DE_STOKES}"
        raise ValueError(msg)
    return 24 / reynolds


def variation_de_la_trainee(avant: float, apres: float) -> float:
    """`(V_2 / V_1)^2 - 1` : la variation relative de la force de trainee."""
    if avant <= 0:
        msg = f"vitesse nulle ou negative : {avant}"
        raise ValueError(msg)
    return float((apres / avant) ** 2 - 1)


def vitesse_de_chute(
    *, masse: float, masse_volumique: float, coefficient: float, surface: float
) -> float:
    """`V = racine(2 m g / (rho C_D A))`, quand la trainee porte le poids."""
    if surface <= 0:
        msg = f"surface nulle ou negative : {surface} m^2"
        raise ValueError(msg)
    return math.sqrt(2 * masse * PESANTEUR / (masse_volumique * coefficient * surface))


def hauteur_de_chute_libre(vitesse: float) -> float:
    """`h = V^2 / (2 g)` : d'ou tomber sans frein pour arriver a cette vitesse."""
    return vitesse**2 / (2 * PESANTEUR)
