"""La variation de masse volumique, rapportee a la masse volumique d'arret.

`regimes.variation_de_masse_volumique` rend `rho_0 / rho - 1`, soit 4,56 %
a `Ma = 0,3`. La planche des classifications parle de la variation de la
masse volumique d'un gaz qui s'ecoule : on la rapporte d'ordinaire a la
valeur d'arret, `(rho_0 - rho) / rho_0`, et c'est elle qui vaut 4,4 % a
`Ma = 0,3`, le chiffre que le cours et la fiche citent.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from polytech.fluides.domain.regimes import GAMMA_DE_L_AIR, MACH_INCOMPRESSIBLE
from polytech.noyau import core

MACH_SONIQUE: Final = 1.0
"""La vitesse du son : `Ma = 1`, par definition du nombre de Mach."""

MACH_HYPERSONIQUE: Final = 5.0
"""Convention usuelle de l'aerodynamique (hors planche) : hypersonique au-dela."""


class RegimeDeMach(Enum):
    """Les quatre domaines que la fiche dessine sur l'axe des Mach."""

    INCOMPRESSIBLE = "incompressible"
    SUBSONIQUE = "subsonique"
    SUPERSONIQUE = "supersonique"
    HYPERSONIQUE = "hypersonique"


def regime_de_mach(mach: float) -> RegimeDeMach:
    """Le domaine de `Ma` : `< 0,3`, `< 1`, `<= 5`, au-dela."""
    if mach < 0:
        msg = f"nombre de Mach negatif : {mach}"
        raise ValueError(msg)
    if mach < MACH_INCOMPRESSIBLE:
        return RegimeDeMach.INCOMPRESSIBLE
    if mach < MACH_SONIQUE:
        return RegimeDeMach.SUBSONIQUE
    return (
        RegimeDeMach.SUPERSONIQUE
        if mach <= MACH_HYPERSONIQUE
        else RegimeDeMach.HYPERSONIQUE
    )


def variation_relative_a_l_arret(mach: float) -> float:
    """`1 - (1 + (gamma - 1) Ma^2 / 2)^(-1/(gamma - 1))`, en ecoulement isentropique."""
    return core.variation_relative_a_l_arret(mach, GAMMA_DE_L_AIR)
