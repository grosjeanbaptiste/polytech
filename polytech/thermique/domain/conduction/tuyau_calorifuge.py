"""Le tuyau calorifuge, et l'echelle des conductivites thermiques.

Planches de conductibilite du cours de thermique. Les conductivites
s'etagent sur quatre decades, des gaz aux metaux ; celle d'un isolant
alveolaire, qui n'est pas un milieu continu, est une conductivite
**apparente**, a peu pres lineaire en temperature autour de l'ambiante.

Un tuyau d'acier calorifuge, parcouru par un fluide chaud, perd par
metre le flux que laissent passer quatre resistances en serie : le
fluide, l'acier, l'isolant, l'air. La conductivite de l'isolant se prend
a la moyenne de ses deux faces, que le flux fixe a son tour : le calcul
se referme par iterations.

Le long du tuyau, le fluide cede ce flux. Tant que la resistance par
metre `R` ne change pas, `q c dt/dx = -(t - t_a) / R` : l'ecart a
l'ambiance decroit exponentiellement, sur la longueur `q c R`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

from polytech.thermique.domain.conduction.parois import (
    Tube,
    conductivite_interpolee,
    flux_lineique_de_tubes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class Etat(Enum):
    """L'etat du materiau, qui ordonne les conductivites."""

    GAZ = "gaz"
    LIQUIDE = "liquide"
    SOLIDE = "solide"


CONDUCTIVITES_TYPIQUES: Final = {
    "air": (Etat.GAZ, 0.025),
    "hydrogene": (Etat.GAZ, 0.17),
    "eau": (Etat.LIQUIDE, 0.6),
    "huile": (Etat.LIQUIDE, 0.15),
    "cuivre": (Etat.SOLIDE, 380.0),
    "aluminium": (Etat.SOLIDE, 210.0),
    "acier": (Etat.SOLIDE, 45.0),
    "verre": (Etat.SOLIDE, 1.0),
}
"""Planche 12 : les valeurs typiques a 20 degres, en W/(m K)."""

ITERATIONS: Final = 50
"""Le point fixe gagne trois chiffres par tour : cinquante suffisent largement."""


@dataclass(frozen=True, slots=True)
class TuyauCalorifuge:
    """Le tube d'acier, le rayon exterieur de l'isolant, sa table, `(K_i, K_e)`."""

    acier: Tube
    rayon_de_l_isolant: float
    table: Sequence[tuple[float, float]]
    echanges: tuple[float, float]

    def _flux(self, isolant: float, fluide: float, ambiance: float) -> float:
        coquille = Tube(self.acier.rayon_exterieur, self.rayon_de_l_isolant, isolant)
        interieur, exterieur = self.echanges
        return flux_lineique_de_tubes(
            (self.acier, coquille),
            fluide,
            ambiance,
            interieur=interieur,
            exterieur=exterieur,
        )

    def _moyenne(self, flux: float, fluide: float, ambiance: float) -> float:
        interieur, exterieur = self.echanges
        cote_fluide = 1 / (2 * math.pi * self.acier.rayon_interieur * interieur)
        chaude = fluide - flux * (cote_fluide + self.acier.resistance())
        froide = ambiance + flux / (2 * math.pi * self.rayon_de_l_isolant * exterieur)
        return (chaude + froide) / 2

    def flux_lineique(self, fluide: float, ambiance: float) -> tuple[float, float]:
        """Le flux par metre, et la conductivite que l'isolant y prend."""
        isolant = conductivite_interpolee(self.table, (fluide + ambiance) / 2)
        for _ in range(ITERATIONS):
            flux = self._flux(isolant, fluide, ambiance)
            moyenne = self._moyenne(flux, fluide, ambiance)
            isolant = conductivite_interpolee(self.table, moyenne)
        return self._flux(isolant, fluide, ambiance), isolant


def temperature_le_long_du_tuyau(
    abscisse: float,
    extremes: tuple[float, float],
    *,
    capacite: float,
    resistance: float,
) -> float:
    """`t_a + (t_e - t_a) exp(-x / (q c R))`, pour `(t_e, t_a)`."""
    if capacite <= 0:
        msg = f"capacite nulle ou negative : {capacite} W/K"
        raise ValueError(msg)
    entree, ambiance = extremes
    amorti = math.exp(-abscisse / (capacite * resistance))
    return ambiance + (entree - ambiance) * amorti
