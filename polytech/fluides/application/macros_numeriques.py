"""Valeurs numeriques du corrige, mises en forme pour siunitx.

Separees de `generation.py`, qui ne fait plus qu'assembler : ce module
portait a lui seul la moitie de ses lignes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import nombre
from polytech.fluides.domain.ecart import ecart_relatif

if TYPE_CHECKING:
    from polytech.fluides.application.resultats import Resultats

_CENT = 100
_MILLE = 1_000


def macros_du_manometre(r: Resultats) -> dict[str, str]:
    """Valeurs numeriques de la partie I."""
    eau, huile, mercure = r.contributions_pa
    return {
        "PressionAtmospheriqueKPa": nombre(
            enonce.PRESSION_ATMOSPHERIQUE.kilopascals,
            1,
        ),
        "ReponseDuCoursKPa": nombre(enonce.REPONSE_DU_COURS.kilopascals, 0),
        "EcartAuCoursPourcent": nombre(
            _CENT
            * ecart_relatif(
                mesure=r.pression_air_kpa,
                reference=enonce.REPONSE_DU_COURS.kilopascals,
            ),
            2,
        ),
        "PressionNiveauMerReferenceKPa": nombre(
            enonce.PRESSION_NIVEAU_MER.kilopascals,
            1,
        ),
        "PressionAirKPa": nombre(r.pression_air_kpa, 2),
        "PressionAirArrondieKPa": nombre(r.pression_air_kpa, 0),
        "PressionRelativeKPa": nombre(r.pression_relative_kpa, 2),
        "ContributionEauPa": nombre(eau, 1),
        "ContributionHuilePa": nombre(huile, 1),
        "ContributionMercurePa": nombre(mercure, 1),
        "ContributionEauKPa": nombre(eau / _MILLE, 3),
        "ContributionHuileKPa": nombre(huile / _MILLE, 3),
        "ContributionMercureKPa": nombre(mercure / _MILLE, 3),
        "PartMercurePourcent": nombre(r.part_mercure_pourcent, 1),
        "SensibiliteMercure": nombre(r.sensibilite_mercure_kpa_par_m, 0),
        "HauteurEauEquivalente": nombre(r.hauteur_eau_equivalente_m, 1),
        "HauteurMercureEquivalente": nombre(
            r.hauteur_mercure_equivalente_m * _CENT,
            0,
        ),
        "PressionNiveauMerKPa": nombre(r.pression_air_niveau_mer_kpa, 1),
        "RapportHauteurValleeV": nombre(r.rapport_hauteur_vallee_v, 3),
    }


def macros_des_champs(r: Resultats) -> dict[str, str]:
    """Valeurs numeriques et litterales de la partie II."""
    return {
        "DivergenceB": r.divergence_b,
        "TauxRotationB": r.taux_rotation_b,
        "ChampAccelerationBX": r.champ_acceleration_b[0],
        "ChampAccelerationBY": r.champ_acceleration_b[1],
        "FonctionCourantC": r.fonction_de_courant_c,
        "PotentielC": r.potentiel_c,
        "AccelerationBX": nombre(r.acceleration_b[0], 0),
        "AccelerationBY": nombre(r.acceleration_b[1], 0),
        "NormeAccelerationB": nombre(r.norme_acceleration_b, 2),
        # Le corrige ecrivait « sqrt(5^2+3^2) = sqrt(34) » a la main.
        "CarreAccelerationB": nombre(
            r.acceleration_b[0] ** 2 + r.acceleration_b[1] ** 2,
            0,
        ),
        "AngleAccelerationB": nombre(r.angle_acceleration_b_degres, 0),
    }
