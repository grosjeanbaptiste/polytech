"""Les ordres de grandeur usuels des coefficients de transmission.

Syllabus de thermique, tome 2, deuxieme partie, section 1.2 (pages 100
et 101), en W/(m^2 K). Ils servent a prevoir, avant tout calcul, le
phenomene **limitatif** : celui dont le coefficient est le plus faible
porte presque toute la resistance, et lui seul demande un calcul precis.
« Cela ne dispense pas de calculer exactement ces coefficients. »

Ce fichier porte une table : une ligne par phenomene, toutes de la meme
forme, `(minimum, maximum)`. Une valeur unique du syllabus y figure deux
fois.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping


class Phenomene(Enum):
    """Les situations que la section 1.2 chiffre."""

    CONVECTION_NATURELLE_DANS_L_AIR = "convection naturelle dans l'air"
    CONVECTION_NATURELLE_DANS_L_EAU = "convection naturelle dans l'eau"
    CONVECTION_NATURELLE_D_UN_ORGANIQUE = "convection naturelle, fluide visqueux"
    EAU_TURBULENTE_EN_TUBE = "eau en ecoulement turbulent, de 1 a 3 m/s"
    HUILE_LAMINAIRE_EN_TUBE = "huile en ecoulement laminaire, vers 0,5 m/s"
    GAZ_TURBULENT_EN_TUBE = "gaz en ecoulement turbulent, de 3 a 10 m/s"
    GAZ_EN_TRAVERS_D_UN_TUBE = "gaz en ecoulement transversal, vers 10 m/s"
    CONDENSATION_DE_LA_VAPEUR = "condensation de la vapeur d'eau"
    CONDENSATION_D_UN_ORGANIQUE = "condensation d'un freon ou d'un organique"
    EBULLITION_DE_L_EAU = "ebullition de l'eau"


ORDRES_DE_GRANDEUR: Final[Mapping[Phenomene, tuple[float, float]]] = {
    Phenomene.CONVECTION_NATURELLE_DANS_L_AIR: (2.0, 5.0),
    Phenomene.CONVECTION_NATURELLE_DANS_L_EAU: (25.0, 500.0),
    Phenomene.CONVECTION_NATURELLE_D_UN_ORGANIQUE: (15.0, 25.0),
    Phenomene.EAU_TURBULENTE_EN_TUBE: (4000.0, 8000.0),
    Phenomene.HUILE_LAMINAIRE_EN_TUBE: (150.0, 150.0),
    Phenomene.GAZ_TURBULENT_EN_TUBE: (15.0, 25.0),
    Phenomene.GAZ_EN_TRAVERS_D_UN_TUBE: (100.0, 100.0),
    Phenomene.CONDENSATION_DE_LA_VAPEUR: (8000.0, 8000.0),
    Phenomene.CONDENSATION_D_UN_ORGANIQUE: (800.0, 1500.0),
    Phenomene.EBULLITION_DE_L_EAU: (15000.0, 50000.0),
}


def phenomene_limitatif(coefficients: Mapping[str, float]) -> str:
    """Le nom du plus faible coefficient : c'est lui qui commande `K_gl`."""
    if not coefficients:
        msg = "aucun coefficient : rien ne limite la transmission"
        raise ValueError(msg)
    return min(coefficients, key=lambda nom: coefficients[nom])
