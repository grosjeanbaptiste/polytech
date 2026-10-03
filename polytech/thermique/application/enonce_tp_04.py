"""Le dimensionnement de la batterie de tubes ailettes du TP 4.

L'air (0,8 kg/s, de 40 a 25 degres) cede sa chaleur a l'eau (1,4 kg/s,
entree a 15 degres) qui circule dans les tubes. Le bilan donne le flux
et la sortie de l'eau ; la moyenne logarithmique, a contre-courant -- la
batterie a plus de quatre rangs --, la conductance `K S` a atteindre.
Le coefficient global rapporte a la surface interieure des tubes donne
alors la longueur de tube, donc le nombre de rangs.

Les proprietes sont prises aux temperatures moyennes des fluides, comme
dans la premiere methode de l'officiel ; la sortie de l'eau, dont la
moyenne depend, est obtenue par point fixe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.thermique.application.proprietes_tp_04 import (
    chaleur_de_l_air,
    chaleur_de_l_eau,
)
from polytech.thermique.domain.echangeurs import (
    efficacite_a_contre_courant,
    moyenne_logarithmique,
)

AIR, ENTREE_AIR, SORTIE_AIR = 0.8, 40.0, 25.0
EAU, ENTREE_EAU = 1.4, 15.0
ITERATIONS, TOLERANCE = 50, 1e-10


def bilan() -> tuple[float, float, float]:
    """La sortie de l'eau, le flux et la conductance `K S` a atteindre."""
    capacite_air = AIR * chaleur_de_l_air((ENTREE_AIR + SORTIE_AIR) / 2)
    flux = capacite_air * (ENTREE_AIR - SORTIE_AIR)
    sortie = ENTREE_EAU
    for _ in range(ITERATIONS):
        nouvelle = ENTREE_EAU + flux / (
            EAU * chaleur_de_l_eau((ENTREE_EAU + sortie) / 2)
        )
        if abs(nouvelle - sortie) < TOLERANCE:
            ecart = moyenne_logarithmique(
                ENTREE_AIR - nouvelle, SORTIE_AIR - ENTREE_EAU
            )
            return nouvelle, flux, flux / ecart
        sortie = nouvelle
    msg = "la sortie de l'eau ne converge pas"
    raise ValueError(msg)


def flux_par_nut(conductance: float) -> float:
    """Le flux que cette conductance echange a contre-courant, par NUT-efficacite."""
    sortie, flux, _ = bilan()
    air = AIR * chaleur_de_l_air((ENTREE_AIR + SORTIE_AIR) / 2)
    eau = flux / (sortie - ENTREE_EAU)
    minimale, maximale = min(air, eau), max(air, eau)
    efficacite = efficacite_a_contre_courant(
        conductance / minimale, minimale / maximale
    )
    return efficacite * minimale * (ENTREE_AIR - ENTREE_EAU)


@dataclass(frozen=True, slots=True)
class Dimensionnement:
    """Chaque etape du calcul de l'officiel, pour une batterie."""

    nusselt_interieur: float
    convection_interieure: float
    reynolds_air: float
    nusselt_air: float
    convection_air: float
    parametre: float
    argument: float
    efficacite_droite: float
    efficacite: float
    coefficient_global: float
    surface_interieure: float
    longueur: float
    rangs_exacts: float

    @property
    def rangs(self) -> int:
        """Le nombre entier de rangs, arrondi au-dessus."""
        return math.ceil(self.rangs_exacts)
