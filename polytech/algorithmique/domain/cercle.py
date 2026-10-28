"""Le disque des programmes 2.3 et 2.8 du syllabus d'introduction.

Les programmes calculent `temp = PI * rayon`, puis la surface `temp *
rayon` et la circonference `2 * temp`, avec la constante `PI` qu'ils
declarent -- le modele en garde l'ordre des operations, qui fixe les
derniers chiffres affiches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Disque:
    """Surface et circonference d'un disque, telles que le programme les calcule."""

    surface: float
    circonference: float


def disque_du_syllabus(pi: float, rayon: float) -> Disque:
    """`temp = pi * rayon` ; surface `temp * rayon`, circonference `2 * temp`."""
    if rayon < 0:
        msg = f"rayon negatif : {rayon}"
        raise ValueError(msg)
    temp = pi * rayon
    return Disque(surface=temp * rayon, circonference=2 * temp)
