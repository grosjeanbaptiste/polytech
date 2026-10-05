"""Les echanges dans une enceinte de surfaces noires.

Syllabus de thermique, tome 1, deuxieme partie, sections 2.1, 2.3.2 et
2.4. Entre deux surfaces noires, l'echange net est proportionnel a
l'ecart des `T^4` et a l'aire d'echange direct, grandeur purement
geometrique et symetrique :

    Phi = 12 sigma (T_1^4 - T_2^4).

Dans une enceinte fermee, le flux net de chaque surface est la somme de
ses echanges avec toutes les autres : un reseau de resistances `1 / ij`
entre des noeuds aux potentiels `sigma T^4`.

Dans une cavite a trois surfaces planes ou convexes, les trois bilans
`S_i = somme ij` suffisent a donner les aires d'echange :
`12 = (S_1 + S_2 - S_3) / 2`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.thermique.domain.rayonnement.corps_noir import emittance

if TYPE_CHECKING:
    from collections.abc import Sequence


def flux_entre_surfaces_noires(
    aire_d_echange: float, premiere: float, seconde: float
) -> float:
    """`12 sigma (T_1^4 - T_2^4)`, en watts ; les temperatures en kelvins."""
    if aire_d_echange < 0:
        msg = f"aire d'echange negative : {aire_d_echange} m^2"
        raise ValueError(msg)
    return aire_d_echange * (emittance(premiere) - emittance(seconde))


def flux_nets_noirs(
    aires: Sequence[Sequence[float]], temperatures: Sequence[float]
) -> list[float]:
    """Le flux net que chaque surface fournit a l'enceinte."""
    indices = range(len(temperatures))
    for i in indices:
        for j in indices:
            if aires[i][j] != aires[j][i]:
                msg = f"aires d'echange non symetriques : {i}, {j}"
                raise ValueError(msg)
    return [
        sum(
            flux_entre_surfaces_noires(aires[i][j], temperatures[i], temperatures[j])
            for j in indices
        )
        for i in indices
    ]


def aire_dans_une_cavite_a_trois(
    premiere: float, seconde: float, troisieme: float
) -> float:
    """`12 = (S_1 + S_2 - S_3) / 2`."""
    aire = (premiere + seconde - troisieme) / 2
    if aire < 0:
        msg = f"ne ferment pas une cavite : {premiere}, {seconde}, {troisieme}"
        raise ValueError(msg)
    return aire


def temperature_d_equilibre(
    flux: float, aire_d_echange: float, ambiance: float
) -> float:
    """La temperature d'une source noire qui evacue `flux` vers l'ambiance."""
    quatrieme = flux / aire_d_echange + emittance(ambiance)
    return float((quatrieme / emittance(1.0)) ** 0.25)
