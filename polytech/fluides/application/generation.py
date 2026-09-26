"""Commande : publier les resultats sous forme de macros LaTeX.

Cote commande du CQRS -- elle produit un effet (l'ecriture des macros) et ne
renvoie aucun resultat de lecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.fluides.application import enonce
from polytech.fluides.application.activite import ResultatsDeLActivite
from polytech.fluides.application.formatage import nombre
from polytech.fluides.application.macros_cinematique import macros_cinematique
from polytech.fluides.application.macros_corroboration import macros_corroboration
from polytech.fluides.application.macros_definitions import macros_definitions
from polytech.fluides.application.macros_derivations import macros_derivations
from polytech.fluides.application.macros_donnees import macros_donnees
from polytech.fluides.application.macros_hypotheses import macros_hypotheses
from polytech.fluides.application.macros_incertitude import macros_incertitude
from polytech.fluides.application.macros_statique import macros_statique
from polytech.fluides.application.macros_validite import macros_validite

_CENT = 100

if TYPE_CHECKING:
    from polytech.fluides.domain.ports import PublicationDeMacros


@dataclass(frozen=True, slots=True)
class GenererMacros:
    """Calcule les resultats puis les publie via le port fourni."""

    publication: PublicationDeMacros

    def executer(self) -> None:
        """Execute la commande ; toute defaillance du port se propage."""
        self.publication.publier(self._macros())

    def _macros(self) -> dict[str, str]:
        r = ResultatsDeLActivite().executer()
        atmospherique = enonce.PRESSION_ATMOSPHERIQUE.kilopascals
        eau, huile, mercure = r.contributions_pa
        return {
            **macros_definitions(),
            **macros_donnees(),
            **macros_derivations(),
            **macros_hypotheses(),
            **macros_incertitude(),
            **macros_statique(),
            **macros_validite(),
            **macros_cinematique(),
            **macros_corroboration(),
            "PressionAtmospheriqueKPa": nombre(atmospherique, 1),
            "ReponseDuCoursKPa": nombre(enonce.REPONSE_DU_COURS.kilopascals, 0),
            "EcartAuCoursPourcent": nombre(
                _CENT
                * abs(r.pression_air_kpa - enonce.REPONSE_DU_COURS.kilopascals)
                / r.pression_air_kpa,
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
            "ContributionEauKPa": nombre(eau / 1_000, 3),
            "ContributionHuileKPa": nombre(huile / 1_000, 3),
            "ContributionMercureKPa": nombre(mercure / 1_000, 3),
            "PartMercurePourcent": nombre(r.part_mercure_pourcent, 1),
            "SensibiliteMercure": nombre(r.sensibilite_mercure_kpa_par_m, 0),
            "HauteurEauEquivalente": nombre(r.hauteur_eau_equivalente_m, 1),
            "HauteurMercureEquivalente": nombre(
                r.hauteur_mercure_equivalente_m * 100,
                0,
            ),
            "PressionNiveauMerKPa": nombre(r.pression_air_niveau_mer_kpa, 1),
            "RapportHauteurValleeV": nombre(r.rapport_hauteur_vallee_v, 3),
            "DivergenceB": r.divergence_b,
            "TauxRotationB": r.taux_rotation_b,
            "ChampAccelerationBX": r.champ_acceleration_b[0],
            "ChampAccelerationBY": r.champ_acceleration_b[1],
            "FonctionCourantC": r.fonction_de_courant_c,
            "PotentielC": r.potentiel_c,
            "AccelerationBX": nombre(r.acceleration_b[0], 0),
            "AccelerationBY": nombre(r.acceleration_b[1], 0),
            "NormeAccelerationB": nombre(r.norme_acceleration_b, 2),
            "AngleAccelerationB": nombre(r.angle_acceleration_b_degres, 0),
        }
