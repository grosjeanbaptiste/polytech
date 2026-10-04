"""La quantite de matiere : de la masse au nombre d'atomes.

Une masse `m` d'un corps de masse molaire `M` contient `n = m / M` moles,
soit `N = n N_A` entites, ou `N_A` est le nombre d'Avogadro. C'est le pont
entre ce que l'on pese et ce que l'on compte -- le syllabus d'electricite
s'en sert pour chiffrer les charges d'un millimetre cube de cuivre
(exemple 1.3.2, page 1-4).
"""

from __future__ import annotations

from typing import Final

CONSTANTE_D_AVOGADRO: Final = 6.02214076e23
"""Nombre d'entites par mole ; exact depuis la redefinition du SI de 2019."""


def quantite_de_matiere(masse: float, masse_molaire: float) -> float:
    """`n = m / M`, en moles ; masse en kg et masse molaire en kg/mol."""
    if masse_molaire <= 0:
        msg = f"masse molaire non strictement positive : {masse_molaire}"
        raise ValueError(msg)
    return masse / masse_molaire


def nombre_d_entites(quantite: float) -> float:
    """`N = n N_A` : le nombre d'atomes, de molecules ou d'ions."""
    return quantite * CONSTANTE_D_AVOGADRO
