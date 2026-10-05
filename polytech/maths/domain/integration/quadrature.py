"""Trois methodes numeriques d'integration : trapezes, Simpson, Gauss.

L'intervalle est coupe en `N` sous-intervalles egaux ; sur chacun :

* **trapezes** -- la moyenne des deux extremites, exacte jusqu'au degre 1 ;
* **Simpson** -- extremites et milieu ponderes 1, 4, 1, exacte jusqu'au
  degre 3 ;
* **Gauss** -- trois points interieurs choisis au mieux (les zeros du
  polynome de Legendre de degre 3), exacte jusqu'au degre 5.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

POINTS_DE_GAUSS = ((-math.sqrt(3 / 5), 5 / 9), (0.0, 8 / 9), (math.sqrt(3 / 5), 5 / 9))
"""Les abscisses et les poids de la formule a trois points, sur `[-1, 1]`."""


def _sous_intervalles(
    debut: float, fin: float, nombre: int
) -> Iterator[tuple[float, float]]:
    if nombre < 1:
        msg = f"au moins un sous-intervalle attendu, {nombre} recu"
        raise ValueError(msg)
    pas = (fin - debut) / nombre
    for k in range(nombre):
        yield debut + k * pas, debut + (k + 1) * pas


def par_les_trapezes(
    fonction: Callable[[float], float], debut: float, fin: float, nombre: int
) -> float:
    """`somme (b - a) (f(a) + f(b)) / 2`."""
    return sum(
        (b - a) * (fonction(a) + fonction(b)) / 2
        for a, b in _sous_intervalles(debut, fin, nombre)
    )


def par_simpson(
    fonction: Callable[[float], float], debut: float, fin: float, nombre: int
) -> float:
    """`somme (b - a) (f(a) + 4 f(m) + f(b)) / 6`, `m` le milieu."""
    return sum(
        (b - a) * (fonction(a) + 4 * fonction((a + b) / 2) + fonction(b)) / 6
        for a, b in _sous_intervalles(debut, fin, nombre)
    )


def par_gauss(
    fonction: Callable[[float], float], debut: float, fin: float, nombre: int
) -> float:
    """La formule de Gauss-Legendre a trois points sur chaque sous-intervalle."""
    return sum(
        (b - a) / 2 * poids * fonction((a + b) / 2 + (b - a) / 2 * abscisse)
        for a, b in _sous_intervalles(debut, fin, nombre)
        for abscisse, poids in POINTS_DE_GAUSS
    )
