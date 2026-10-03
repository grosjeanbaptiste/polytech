"""Macros du TP 4 de thermique : la batterie de tubes ailettes.

Le bilan, la conductance a atteindre, chaque etape du dimensionnement de
la batterie de l'officiel, puis les quatre batteries cote a cote -- le
choix se justifie sur toutes. Et ce que donnent reellement les rangs
entiers : arrondir au-dessus surdimensionne, et la methode NUT dit de
combien l'air sort plus froid que demande.

Le prefixe est `Echangeur`.
"""

from __future__ import annotations

import math

from polytech.thermique.application import enonce_tp_04 as e
from polytech.thermique.application.dimensionnement_tp_04 import dimensionner
from polytech.thermique.application.formatage import nombre
from polytech.thermique.application.proprietes_tp_04 import (
    BATTERIES,
    DIAMETRE_INTERIEUR,
    LONGUEUR_DE_TUBE,
    chaleur_de_l_air,
)

RANGS = ("Un", "Deux", "Trois", "Quatre")


def _macros_du_bilan() -> dict[str, str]:
    sortie, flux, conductance = e.bilan()
    return {
        "EchangeurSortieEau": nombre(sortie, 3),
        "EchangeurFlux": nombre(flux, 0),
        "EchangeurEcartLog": nombre(flux / conductance, 3),
        "EchangeurConductance": nombre(conductance, 2),
        "EchangeurNut": nombre(e.flux_par_nut(conductance), 0),
    }


def _macros_de_l_officielle() -> dict[str, str]:
    d = dimensionner("TCPL16-40-4-50")
    batterie = BATTERIES["TCPL16-40-4-50"]
    reelle = d.coefficient_global * (
        d.rangs
        * batterie.tubes_par_rang
        * LONGUEUR_DE_TUBE
        * math.pi
        * DIAMETRE_INTERIEUR
    )
    capacite_air = e.AIR * chaleur_de_l_air((e.ENTREE_AIR + e.SORTIE_AIR) / 2)
    return {
        "EchangeurNuInterieur": nombre(d.nusselt_interieur, 2),
        "EchangeurKi": nombre(d.convection_interieure, 0),
        "EchangeurReAir": nombre(d.reynolds_air, 0),
        "EchangeurNuAir": nombre(d.nusselt_air, 2),
        "EchangeurKe": nombre(d.convection_air, 1),
        "EchangeurM": nombre(d.parametre, 2),
        "EchangeurArgument": nombre(d.argument, 4),
        "EchangeurOmegaL": nombre(d.efficacite_droite, 4),
        "EchangeurOmega": nombre(d.efficacite, 3),
        "EchangeurKgl": nombre(d.coefficient_global, 1),
        "EchangeurSi": nombre(d.surface_interieure, 4),
        "EchangeurLongueur": nombre(d.longueur, 2),
        "EchangeurRangsExacts": nombre(d.rangs_exacts, 3),
        "EchangeurRangs": str(d.rangs),
        "EchangeurSortieAirReelle": nombre(
            e.ENTREE_AIR - e.flux_par_nut(reelle) / capacite_air, 2
        ),
    }


def _macros_des_quatre() -> dict[str, str]:
    macros: dict[str, str] = {}
    for rang, nom in zip(RANGS, BATTERIES, strict=True):
        d = dimensionner(nom)
        tube = d.rangs_exacts * BATTERIES[nom].tubes_par_rang * LONGUEUR_DE_TUBE
        macros |= {
            f"EchangeurRe{rang}": nombre(d.reynolds_air, 0),
            f"EchangeurKe{rang}": nombre(d.convection_air, 1),
            f"EchangeurOmega{rang}": nombre(d.efficacite, 3),
            f"EchangeurKgl{rang}": nombre(d.coefficient_global, 0),
            f"EchangeurExacts{rang}": nombre(d.rangs_exacts, 2),
            f"EchangeurEntiers{rang}": str(d.rangs),
            f"EchangeurTube{rang}": nombre(tube, 1),
        }
    return macros


def macros_du_tp_quatre() -> dict[str, str]:
    """Le bilan, la batterie de l'officiel, les quatre batteries."""
    return {**_macros_du_bilan(), **_macros_de_l_officielle(), **_macros_des_quatre()}
