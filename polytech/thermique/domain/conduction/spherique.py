"""La conduction a travers une coque spherique.

L'erreur classique est de transposer le cas plan. En plan la temperature
est affine dans l'epaisseur ; en spherique la surface traversee croit
comme r^2, donc le gradient decroit comme 1/r^2 et le profil varie comme
1/r. C'est la conservation du flux qui l'impose, et non la loi de Fourier
seule :

    d/dr ( r^2 dT/dr ) = 0   =>   T(r) = -C_1/r + C_2

d'ou une resistance qui n'est pas `e / (lambda S)` mais
`(1/r_i - 1/r_e) / (4 pi lambda)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.thermique.domain.conduction.symboles import CONSTANTES, RAYON
from polytech.thermique.domain.echanges import ResistanceThermique

if TYPE_CHECKING:
    from polytech.thermique.domain.geometrie import Rayon
    from polytech.thermique.domain.materiaux import Conductivite

TEMPERATURE_SPHERIQUE = sp.Function("T")
"""La temperature comme fonction du rayon, avant integration."""


@dataclass(frozen=True, slots=True)
class CoqueSpherique:
    """Coque spherique homogene, entre deux rayons."""

    conductivite: Conductivite
    interieur: Rayon
    exterieur: Rayon

    def __post_init__(self) -> None:
        """Refuse la coque inversee : sa resistance serait negative."""
        if self.exterieur.metres <= self.interieur.metres:
            msg = (
                "rayon exterieur plus petit que l'interieur : "
                f"{self.exterieur.metres} <= {self.interieur.metres}"
            )
            raise ValueError(msg)

    def loi_de_conservation(self) -> sp.Eq:
        """Le flux total ne depend pas du rayon : d/dr (r^2 dT/dr) = 0."""
        gradient = sp.Derivative(TEMPERATURE_SPHERIQUE(RAYON), RAYON)
        return sp.Eq(sp.Derivative(RAYON**2 * gradient, RAYON), 0)

    def profil_general(self) -> sp.Expr:
        """T(r) = -C_1/r + C_2, solution de la conservation du flux."""
        premiere, seconde = CONSTANTES
        return -premiere / RAYON + seconde

    def resistance(self) -> ResistanceThermique:
        """(1/r_i - 1/r_e) / (4 pi lambda), en K/W."""
        ecart = 1.0 / self.interieur.metres - 1.0 / self.exterieur.metres
        return ResistanceThermique(
            kelvins_par_watt=ecart / (4.0 * math.pi * self.conductivite.watts_par_m_k),
        )
