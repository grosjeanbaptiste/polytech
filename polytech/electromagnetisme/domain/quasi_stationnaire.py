"""L'hypothese des etats quasi-stationnaires.

Le syllabus d'electricite generale (15.1). Si les charges et les courants
d'un systeme changent peu pendant le temps `L / c_0` qu'une perturbation
met a parcourir sa plus grande dimension utile, on peut negliger le
retard des potentiels : le champ est a chaque instant celui de l'etat
stationnaire, au champ electrique induit pres. Pour des phenomenes
sinusoidaux, la condition est que `L` soit petit devant la longueur
d'onde `lambda = c_0 T` -- 6 000 km a 50 Hz.

L'hypothese revient a negliger la propagation, donc les proprietes
magnetisantes du courant de deplacement : le courant vrai est
conservatif, et le systeme ne rayonne pas.
"""

from __future__ import annotations

from polytech.electromagnetisme.domain.onde import (
    celerite_dans_le_vide,
    longueur_d_onde,
)


def rapport_a_la_longueur_d_onde(dimension: float, frequence: float) -> float:
    """`L / lambda` : petit devant un, l'etat est quasi-stationnaire."""
    if dimension <= 0:
        msg = f"dimension nulle ou negative : {dimension} m"
        raise ValueError(msg)
    return dimension / longueur_d_onde(celerite_dans_le_vide(), frequence)
