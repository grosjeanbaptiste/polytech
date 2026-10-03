"""Macros du TP 2 de thermique : le camion refrigere et le four.

Camion : les quatorze peintures de la fiche technique, chacune dans les
deux situations de l'enonce, pour que le choix se justifie sur toute la
fiche et pas sur deux lignes. Les deux peintures que l'officiel compare
ont en plus leur `K_ray` linearise.

Four : les aires d'echange, l'echange gris, la puissance, la vitesse de
chauffe, les radiosites, la temperature des parois, et le nombre de Biot
qui autorise la tole a temperature uniforme.

Les prefixes sont `Camion` et `Four`.
"""

from __future__ import annotations

from polytech.thermique.application import enonce_tp_02 as e
from polytech.thermique.application.formatage import nombre

RANGS = (
    "Un",
    "Deux",
    "Trois",
    "Quatre",
    "Cinq",
    "Six",
    "Sept",
    "Huit",
    "Neuf",
    "Dix",
    "Onze",
    "Douze",
    "Treize",
    "Quatorze",
)
SITUATIONS = {"Arret": 15.0, "Route": 60.0}
"""Vent de 5 m/s a l'arret, 90 km/h en roulant : les deux `K_cf` de l'enonce."""


def _macros_d_une_peinture(rang: str, nom: str) -> dict[str, str]:
    absorptivite, emissivite = e.PEINTURES[nom]
    macros = {
        f"CamionAlpha{rang}": nombre(absorptivite, 2),
        f"CamionEps{rang}": nombre(emissivite, 2),
    }
    for situation, convection in SITUATIONS.items():
        toit = e.toit_du_camion(nom, convection)
        macros[f"CamionTemp{situation}{rang}"] = nombre(
            toit.temperature() - e.KELVIN, 1
        )
        macros[f"CamionCharge{situation}{rang}"] = nombre(
            toit.flux_vers_l_interieur(e.SURFACE_DU_TOIT), 0
        )
    return macros


def _macros_du_camion() -> dict[str, str]:
    macros = {
        "CamionConductance": nombre(e.conductance_interieure(), 3),
        "CamionResistanceIsolant": nombre(e.EPAISSEUR / e.CONDUCTIVITE, 3),
        "CamionKrayBlanc": nombre(
            e.toit_du_camion("AZW/LA-II", 15.0).coefficient_de_rayonnement(), 2
        ),
        "CamionKrayNoir": nombre(
            e.toit_du_camion("MLS-85-SB-C", 15.0).coefficient_de_rayonnement(), 2
        ),
    }
    for rang, nom in zip(RANGS, e.PEINTURES, strict=True):
        macros |= _macros_d_une_peinture(rang, nom)
    return macros


def _macros_du_four() -> dict[str, str]:
    enceinte = e.enceinte_du_four()
    s12, s13, s23 = e.aires_du_four()
    j1, j2 = enceinte.radiosites()
    flux_surfacique = enceinte.flux() / (2 * e.LARGEUR_TOLE)
    coefficient = flux_surfacique / (e.RESISTANCES - e.TOLE_INITIALE)
    return {
        "FourEmissiviteTole": nombre(e.EMISSIVITE_TOLE, 1),
        "FourEmissiviteParois": nombre(e.EMISSIVITE_PAROIS, 1),
        "FourSUnDeux": nombre(s12, 4),
        "FourSUnTrois": nombre(s13, 4),
        "FourSDeuxTrois": nombre(s23, 4),
        "FourNoir": nombre(enceinte.echange_noir(), 3),
        "FourGris": nombre(enceinte.echange_gris(), 3),
        "FourPuissance": nombre(enceinte.flux(), 0),
        "FourVitesse": nombre(e.vitesse_de_chauffe(), 3),
        "FourJUn": nombre(j1, 0),
        "FourJDeux": nombre(j2, 0),
        "FourParois": nombre(enceinte.temperature_de_la_paroi() - e.KELVIN, 1),
        "FourEcartOfficiel": nombre(
            enceinte.temperature_de_la_paroi() - e.KELVIN - e.PAROIS_SELON_L_OFFICIEL, 1
        ),
        "FourCoefficient": nombre(coefficient, 1),
        "FourBiot": nombre(coefficient * e.EPAISSEUR_TOLE / 2 / e.CONDUCTIVITE_TOLE, 4),
    }


def macros_du_tp_deux() -> dict[str, str]:
    """Les deux exercices du TP."""
    return {**_macros_du_camion(), **_macros_du_four()}
