"""La pesanteur : la force que subit toute masse a la surface de la Terre.

« La force de pesanteur ne depend plus que de la masse `m` de l'objet et
s'ecrit `P = m g`, ou `g` est l'acceleration de la pesanteur, dirigee
vers le bas et egale a 9.81 m/s2 sous notre latitude. »

La constante vit ici, dans la discipline qui l'enseigne ; la mecanique
des fluides, qui pese une colonne de liquide, l'importe.

Source : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
section 1.7.4, page 23, relation (1.32).
"""

from __future__ import annotations

from typing import Final

PESANTEUR: Final = 9.81
"""Acceleration de la pesanteur retenue par les cours, en m/s^2."""


def poids(masse: float) -> float:
    """`P = m g`, en newtons ; masse en kilogrammes."""
    if masse < 0:
        msg = f"masse negative : {masse}"
        raise ValueError(msg)
    return masse * PESANTEUR
