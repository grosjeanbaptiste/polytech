"""Les aires d'echange direct en deux dimensions, par les cordes croisees.

Pour deux surfaces planes d'une geometrie longue -- un four, un canal --
la methode de Hottel donne l'aire d'echange par metre de longueur sans
aucune integrale : la demi-difference entre la somme des **cordes
croisees** et celle des **cordes non croisees**, tendues entre les
extremites des deux surfaces.

Une corde est un fil tendu : si un obstacle coupe la ligne droite, le fil
s'enroule sur ses bords, et sa longueur est celle de la ligne brisee qui
passe par eux. `corde_tendue` prend donc ses points de passage dans
l'ordre, et c'est a l'appelant -- qui voit la geometrie -- de les donner.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

Point = tuple[float, float]


def corde_tendue(*points: Point) -> float:
    """La longueur de la ligne brisee qui passe par ces points, dans l'ordre."""
    return sum(
        math.dist(depart, arrivee) for depart, arrivee in itertools.pairwise(points)
    )


def corde_croisee_de_deux_tubes(diametre: float, pas: float) -> float:
    """Le fil croise tendu entre deux tubes de meme diametre, au pas donne.

    Un segment tangent aux deux, plus l'arc dont il s'enroule sur chacun.
    """
    if pas < diametre:
        msg = f"tubes qui se chevauchent : pas {pas} m, diametre {diametre} m"
        raise ValueError(msg)
    tangente = math.sqrt(pas**2 - diametre**2)
    return tangente + diametre * math.asin(diametre / pas)


def aire_par_les_cordes(
    croisees: Sequence[float], non_croisees: Sequence[float]
) -> float:
    """`(somme croisees - somme non croisees) / 2` ; leve si elle est negative."""
    aire = (sum(croisees) - sum(non_croisees)) / 2
    if aire < 0:
        msg = (
            f"aire negative ({aire:.4g}) : cordes croisees et non croisees "
            "probablement echangees"
        )
        raise ValueError(msg)
    return aire
