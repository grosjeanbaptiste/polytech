"""Requete : calculer tous les resultats de l'activite d'integration 1.

Cote requete du CQRS -- sans effet de bord, sans ecriture. La commande de
generation vit dans :mod:`fluides.application.generation`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex
from polytech.fluides.application.resultats import Resultats
from polytech.fluides.domain.grandeurs import PESANTEUR, Hauteur
from polytech.fluides.domain.statique.parements import (
    ParementTriangulaire,
    ParementVertical,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from polytech.fluides.domain.statique.efforts import Parement

_HAUTEUR_REFERENCE = Hauteur(metres=10.0)
_MOITIE = 0.5


class ResultatsDeLActivite:
    """Assemble les reponses des deux parties de l'activite."""

    def executer(self) -> Resultats:
        """Calcule et renvoie l'integralite des resultats du corrige."""
        air = enonce.MANOMETRE.pression_amont(aval=enonce.PRESSION_ATMOSPHERIQUE)
        relative = air.relative_a(enonce.PRESSION_ATMOSPHERIQUE)
        contributions = tuple(c.pascals for c in enonce.MANOMETRE.contributions())
        ax, ay = enonce.CHAMP_B.acceleration_en(*enonce.POINT_B)
        mur, vallee = self._parements()
        return Resultats(
            pression_air_kpa=air.kilopascals,
            pression_relative_kpa=relative.kilopascals,
            contributions_pa=contributions,  # type: ignore[arg-type]
            part_mercure_pourcent=100 * enonce.MANOMETRE.part_dominante(),
            sensibilite_mercure_kpa_par_m=enonce.MERCURE.kg_par_m3 * PESANTEUR / 1_000,
            hauteur_eau_equivalente_m=relative.pascals
            / (enonce.EAU.kg_par_m3 * PESANTEUR),
            hauteur_mercure_equivalente_m=relative.pascals
            / (enonce.MERCURE.kg_par_m3 * PESANTEUR),
            pression_air_niveau_mer_kpa=(101_300.0 + relative.pascals) / 1_000,
            rapport_poussee_prismatique=self._rapport(mur, mur.poussee),
            rapport_moment_prismatique=self._rapport(mur, mur.moment),
            rapport_hauteur_vallee_v=vallee.hauteur_pour_fraction_de_volume(_MOITIE),
            rapport_poussee_vallee_v=self._rapport(vallee, vallee.poussee),
            divergence_b=expression_latex(enonce.CHAMP_B.divergence()),
            taux_rotation_b=expression_latex(enonce.CHAMP_B.taux_de_rotation()),
            champ_acceleration_b=tuple(  # type: ignore[arg-type]
                expression_latex(c) for c in enonce.CHAMP_B.acceleration()
            ),
            acceleration_b=(ax, ay),
            norme_acceleration_b=math.hypot(ax, ay),
            angle_acceleration_b_degres=math.degrees(math.atan2(ay, ax)),
            fonction_de_courant_c=expression_latex(
                enonce.CHAMP_C.fonction_de_courant(),
            ),
            potentiel_c=expression_latex(enonce.CHAMP_C.potentiel_des_vitesses()),
        )

    @staticmethod
    def _parements() -> tuple[ParementVertical, ParementTriangulaire]:
        return (
            ParementVertical(fluide=enonce.EAU, largeur=1.0),
            ParementTriangulaire(fluide=enonce.EAU, pente=0.5),
        )

    @staticmethod
    def _rapport(
        parement: Parement,
        grandeur: Callable[[Hauteur], float],
    ) -> float:
        """Rapport de ``grandeur`` entre volume halve et volume plein."""
        reduite = _HAUTEUR_REFERENCE.metres * parement.hauteur_pour_fraction_de_volume(
            _MOITIE,
        )
        return grandeur(Hauteur(metres=reduite)) / grandeur(_HAUTEUR_REFERENCE)
