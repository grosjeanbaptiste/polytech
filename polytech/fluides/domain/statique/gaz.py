"""Cours II -- de quoi chiffrer l'hypothese « le poids de l'air est negligeable ».

Le corrige invoquait une masse volumique d'air d'environ
1,5 kg/m3 sans dire d'ou elle venait. Elle se deduit de la loi des gaz
parfaits, et l'erreur commise en negligeant la colonne gazeuse se chiffre
dans la foulee : c'est ce qui transforme une hypothese affirmee en une
hypothese justifiee.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.grandeurs import (
    PESANTEUR,
    Hauteur,
    MasseVolumique,
    Pression,
)

CONSTANTE_DES_GAZ = 8.314
"""Constante molaire des gaz parfaits, en J/(mol K).

Valeur CODATA 2018 arrondie (8,314 462 618... exactement depuis la
redefinition du SI de 2019).
"""

TEMPERATURE_AMBIANTE = 293.15
"""Temperature de reference, en kelvins (20 degres Celsius).

L'enonce ne donne aucune temperature : c'est une hypothese de travail. Le
module fournit de quoi montrer que la conclusion n'en depend pas
(cf. :func:`majorant_de_la_part_negligee`).
"""


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
"""Air sec, masse molaire moyenne 29 g/mol (valeur normalisee : 28,96)."""


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


def part_negligee_litterale() -> sp.Expr:
    """Forme litterale de la part negligee : ``M g h / (R T)``.

    Obtenue en reportant ``rho = p M / (R T)`` dans ``rho g h / p``. La
    pression se simplifie : la part negligee n'en depend pas, et justifier
    (H3) au moyen de la pression cherchee n'est donc pas circulaire.
    """
    pression, masse_molaire = (
        sp.Symbol("p", positive=True),
        sp.Symbol("M", positive=True),
    )
    gaz, temperature = sp.Symbol("R", positive=True), sp.Symbol("T", positive=True)
    pesanteur, hauteur = sp.Symbol("g", positive=True), sp.Symbol("h", positive=True)
    masse_volumique = pression * masse_molaire / (gaz * temperature)
    return sp.simplify(masse_volumique * pesanteur * hauteur / pression)


def part_negligee_independante_de_la_pression(
    gaz: GazParfait,
    hauteur: Hauteur,
    temperature: float,
    pesanteur: float = PESANTEUR,
) -> float:
    """Evalue ``M g h / (R T)``, sans jamais mentionner de pression."""
    if temperature <= 0:
        msg = f"temperature non strictement positive : {temperature}"
        raise ValueError(msg)
    return (
        gaz.masse_molaire
        * pesanteur
        * hauteur.metres
        / (CONSTANTE_DES_GAZ * temperature)
    )


def majorant_de_la_part_negligee(
    gaz: GazParfait,
    hauteur_maximale: Hauteur,
    temperature_minimale: float,
    pesanteur: float = PESANTEUR,
) -> float:
    """Pire cas sur toute une plage : ciel gazeux le plus haut, gaz le plus froid.

    La part negligee croit avec la hauteur et decroit avec la temperature :
    le majorant s'obtient donc en prenant les deux bornes defavorables.
    """
    return part_negligee_independante_de_la_pression(
        gaz,
        hauteur_maximale,
        temperature_minimale,
        pesanteur,
    )
