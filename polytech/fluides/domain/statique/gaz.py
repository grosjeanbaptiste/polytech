"""Cours II -- de quoi chiffrer l'hypothese « le poids de l'air est negligeable ».

Le corrige invoquait une masse volumique d'air d'environ
1,5 kg/m3 sans dire d'ou elle venait. Elle se deduit de la loi des gaz
parfaits, et l'erreur commise en negligeant la colonne gazeuse se chiffre
dans la foulee : c'est ce qui transforme une hypothese affirmee en une
hypothese justifiee.
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.fluides.domain.grandeurs import (
    PESANTEUR,
    Hauteur,
    MasseVolumique,
    Pression,
)

CONSTANTE_DES_GAZ = 8.314
"""Constante molaire des gaz parfaits, en J/(mol K)."""

TEMPERATURE_AMBIANTE = 293.15
"""Temperature de reference, en kelvins (20 degres Celsius)."""


@dataclass(frozen=True, slots=True)
class GazParfait:
    """Gaz caracterise par sa seule masse molaire."""

    masse_molaire: float
    nom: str = "gaz"

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.masse_molaire <= 0:
            msg = f"masse molaire non strictement positive : {self.masse_molaire}"
            raise ValueError(msg)

    def masse_volumique(
        self,
        pression: Pression,
        temperature: float,
    ) -> MasseVolumique:
        """``rho = p M / (R T)``, tiree de ``p V = n R T``."""
        if temperature <= 0:
            msg = f"temperature non strictement positive : {temperature}"
            raise ValueError(msg)
        return MasseVolumique(
            kg_par_m3=pression.pascals
            * self.masse_molaire
            / (CONSTANTE_DES_GAZ * temperature),
        )


AIR: GazParfait = GazParfait(masse_molaire=0.029, nom="air")
"""Air sec, masse molaire moyenne 29 g/mol."""


def part_negligee(
    gaz: GazParfait,
    pression: Pression,
    hauteur: Hauteur,
    temperature: float,
    pesanteur: float = PESANTEUR,
) -> float:
    """Poids de la colonne gazeuse rapporte a la pression mesuree.

    C'est l'erreur relative que l'on commet en supposant la pression
    uniforme dans le ciel gazeux du reservoir.
    """
    colonne = gaz.masse_volumique(pression, temperature).kg_par_m3
    return colonne * pesanteur * hauteur.metres / pression.pascals
