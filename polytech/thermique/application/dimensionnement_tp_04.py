"""Le dimensionnement d'une batterie, etape par etape, comme l'officiel.

Cote eau, la relation de Colburn dans chaque tube ; cote air, le
polynome de la batterie, sur la section frontale de la gaine et
l'espacement des ailettes ; l'efficacite d'ailette circulaire, depuis
l'ailette droite de meme hauteur ; puis le coefficient global rapporte a
la surface interieure des tubes, et la longueur de tube qu'il faut.
"""

from __future__ import annotations

import math

import numpy as np

from polytech.thermique.application.enonce_tp_04 import (
    AIR,
    EAU,
    ENTREE_AIR,
    ENTREE_EAU,
    SORTIE_AIR,
    Dimensionnement,
    bilan,
)
from polytech.thermique.application.proprietes_tp_04 import (
    BATTERIES,
    CONDUCTIVITE_AILETTE,
    CONDUCTIVITE_TUBE,
    DIAMETRE_EXTERIEUR,
    DIAMETRE_INTERIEUR,
    LONGUEUR_DE_TUBE,
    Batterie,
    chaleur_de_l_eau,
    conductivite_de_l_air,
    conductivite_de_l_eau,
    viscosite_de_l_air,
    viscosite_de_l_eau,
)
from polytech.thermique.domain.conduction.ailettes import (
    AiletteCirculaire,
    AiletteDroite,
)


def _cote_eau(batterie: Batterie, moyenne: float) -> tuple[float, float]:
    """Le Nusselt de Colburn dans un tube, et le coefficient qu'il donne."""
    viscosite = viscosite_de_l_eau(moyenne)
    reynolds = (
        4 * (EAU / batterie.tubes_par_rang) / (math.pi * DIAMETRE_INTERIEUR * viscosite)
    )
    prandtl = viscosite * chaleur_de_l_eau(moyenne) / conductivite_de_l_eau(moyenne)
    nusselt = 0.0234 * reynolds**0.8 * prandtl ** (1 / 3)
    return nusselt, nusselt * conductivite_de_l_eau(moyenne) / DIAMETRE_INTERIEUR


def _cote_air(batterie: Batterie, moyenne: float) -> tuple[float, float, float]:
    """Le Reynolds frontal, le Nusselt du polynome, le coefficient."""
    frontale = batterie.tubes_par_rang * batterie.cote * LONGUEUR_DE_TUBE
    reynolds = AIR / frontale * batterie.pas / viscosite_de_l_air(moyenne)
    nusselt = float(np.polyval(batterie.nusselt, reynolds))
    libre = batterie.pas - batterie.epaisseur
    return reynolds, nusselt, nusselt * conductivite_de_l_air(moyenne) / libre


def dimensionner(nom: str) -> Dimensionnement:
    """Toutes les etapes pour cette batterie ; leve si elle n'est pas proposee."""
    batterie = BATTERIES[nom]
    sortie, _, conductance = bilan()
    nu_i, k_i = _cote_eau(batterie, (ENTREE_EAU + sortie) / 2)
    re_e, nu_e, k_e = _cote_air(batterie, (ENTREE_AIR + SORTIE_AIR) / 2)
    parametre = math.sqrt(2 * k_e / (CONDUCTIVITE_AILETTE * batterie.epaisseur))
    base, externe = DIAMETRE_EXTERIEUR / 2, batterie.cote / math.sqrt(math.pi)
    ailette = AiletteCirculaire(AiletteDroite(parametre, externe - base), base, externe)
    efficacite = ailette.efficacite()
    exterieure = k_e * (batterie.primaire + efficacite * batterie.ailettes)
    paroi = (
        DIAMETRE_INTERIEUR
        / 2
        / CONDUCTIVITE_TUBE
        * math.log(DIAMETRE_EXTERIEUR / DIAMETRE_INTERIEUR)
    )
    coefficient = 1 / (1 / k_i + paroi + batterie.interieure / exterieure)
    surface = conductance / coefficient
    longueur = surface / (math.pi * DIAMETRE_INTERIEUR)
    return Dimensionnement(
        nusselt_interieur=nu_i,
        convection_interieure=k_i,
        reynolds_air=re_e,
        nusselt_air=nu_e,
        convection_air=k_e,
        parametre=parametre,
        argument=ailette.droite.argument,
        efficacite_droite=ailette.droite.efficacite(),
        efficacite=efficacite,
        coefficient_global=coefficient,
        surface_interieure=surface,
        longueur=longueur,
        rangs_exacts=longueur / (batterie.tubes_par_rang * LONGUEUR_DE_TUBE),
    )
