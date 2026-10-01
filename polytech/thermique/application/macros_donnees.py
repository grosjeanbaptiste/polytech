"""Macros des donnees de l'enonce du TP 1.

Le document ne tape aucun chiffre : il cite ces macros. Le facteur
d'absorption et le flux solaire restent separes, comme l'enonce les donne,
et leur produit est publie a part pour que le lecteur voie les trois.
"""

from __future__ import annotations

from polytech.thermique.application import enonce
from polytech.thermique.application.formatage import nombre, scientifique


def macros_donnees_charbon() -> dict[str, str]:
    """Les dix donnees de l'exercice 1, telles que l'enonce les ecrit."""
    return {
        "EpaisseurCharbon": nombre(enonce.EPAISSEUR_CHARBON.metres, 0),
        "GenerationCharbon": nombre(enonce.GENERATION_CHARBON.watts_par_m3, 0),
        "ConductiviteCharbon": nombre(enonce.CONDUCTIVITE_CHARBON.watts_par_m_k, 1),
        "FluxSolaire": nombre(enonce.FLUX_SOLAIRE.watts_par_m2, 0),
        "AbsorptionCharbon": nombre(enonce.ABSORPTION_CHARBON, 2),
        "FluxAbsorbe": nombre(enonce.FLUX_ABSORBE.watts_par_m2, 1),
        "TemperatureAir": nombre(enonce.TEMPERATURE_AIR.celsius, 0),
        "EchangeExterieur": nombre(enonce.ECHANGE_EXTERIEUR.watts_par_m2_k, 0),
        "TemperatureSol": nombre(enonce.TEMPERATURE_SOL.celsius, 0),
        "EchangeSol": nombre(enonce.ECHANGE_SOL.watts_par_m2_k, 0),
    }


def macros_donnees_reservoir() -> dict[str, str]:
    """Les sept donnees de l'exercice 2, et les deux rayons qui s'en deduisent."""
    return {
        "DiametreInterieur": nombre(enonce.DIAMETRE_INTERIEUR, 1),
        "EpaisseurSilice": nombre(enonce.EPAISSEUR_SILICE.metres * 1_000, 0),
        "ConductiviteSilice": nombre(enonce.CONDUCTIVITE_SILICE.watts_par_m_k, 4),
        "TemperatureHydrogene": nombre(enonce.TEMPERATURE_HYDROGENE.kelvins, 0),
        "TemperatureAmbiante": nombre(enonce.TEMPERATURE_AMBIANTE.kelvins, 0),
        "EchangeReservoir": nombre(enonce.ECHANGE_RESERVOIR.watts_par_m2_k, 0),
        "ChaleurLatente": scientifique(enonce.CHALEUR_LATENTE.joules_par_kg, 1),
        "RayonInterieur": nombre(enonce.COQUE.interieur.metres, 3),
        "RayonExterieur": nombre(enonce.COQUE.exterieur.metres, 3),
    }
