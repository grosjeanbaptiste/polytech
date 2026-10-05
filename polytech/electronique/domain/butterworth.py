"""Le filtre de Butterworth : module, parametre d'ondulation, poles.

Chapitre 3, partie I, planches 27 a 31. Un Butterworth n'ondule nulle
part : `|T(j w)| = 1 / racine(1 + eps^2 (w / w_p)^(2N))`. Le parametre
`eps` ne depend que de l'attenuation toleree au bord de la bande ; l'ordre
`N` se cherche ensuite, en l'incrementant jusqu'a tenir la bande
bloquante.

Ses `N` poles sont sur un demi-cercle de rayon `w_p / eps^(1/N)`, a
gauche de l'axe imaginaire, ecartes de `pi / N`.
"""

from __future__ import annotations

import cmath
import math


def ondulation_depuis_l_attenuation(attenuation: float) -> float:
    """`eps = racine(1 / (1 - A_max)^2 - 1)`, pour `A_max` en echelle lineaire."""
    if not 0 <= attenuation < 1:
        msg = f"attenuation hors de [0, 1[ : {attenuation}"
        raise ValueError(msg)
    return math.sqrt(1 / (1 - attenuation) ** 2 - 1)


def _exiger_un_ordre(ordre: int) -> None:
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)


def module_de_butterworth(
    pulsation: float, ordre: int, epsilon: float = 1.0, *, passante: float = 1.0
) -> float:
    """`1 / racine(1 + eps^2 (w / w_p)^(2N))`."""
    _exiger_un_ordre(ordre)
    if passante <= 0:
        msg = f"pulsation passante nulle ou negative : {passante} rad/s"
        raise ValueError(msg)
    return 1 / math.sqrt(1 + epsilon**2 * (pulsation / passante) ** (2 * ordre))


def poles_de_butterworth(
    ordre: int, *, epsilon: float = 1.0, passante: float = 1.0
) -> tuple[complex, ...]:
    """`p_k = (w_p / eps^(1/N)) exp(j [pi/2 + pi/(2N) + (k - 1) pi/N])`."""
    _exiger_un_ordre(ordre)
    if epsilon <= 0:
        msg = f"parametre d'ondulation nul ou negatif : {epsilon}"
        raise ValueError(msg)
    rayon = passante / epsilon ** (1 / ordre)
    depart = math.pi / 2 + math.pi / (2 * ordre)
    return tuple(
        rayon * cmath.exp(1j * (depart + rang * math.pi / ordre))
        for rang in range(ordre)
    )
