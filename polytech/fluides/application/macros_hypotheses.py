"""Macros qui *chiffrent* les hypotheses au lieu de les affirmer.

Le corrige posait « le poids de l'air est negligeable » avec une masse
volumique sortie de nulle part, et annoncait que les trois familles de
courbes coincident sans jamais calculer la troisieme. Ces macros comblent
les deux manques.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex, nombre
from polytech.fluides.domain.cinematique.emission import (
    invariant_de_la_ligne_emission,
    ligne_emission,
)
from polytech.fluides.domain.grandeurs import Hauteur, Pression
from polytech.fluides.domain.statique.gaz import (
    AIR,
    TEMPERATURE_AMBIANTE,
    majorant_de_la_part_negligee,
    part_negligee,
    part_negligee_litterale,
)
from polytech.fluides.domain.statique.stevin import variation_de_pression

CIEL_GAZEUX = Hauteur(metres=0.5)
"""Hauteur d'air supposee au-dessus de l'eau dans le reservoir."""

CIEL_GAZEUX_MAJORANT = Hauteur(metres=2.0)
"""Borne haute plausible du ciel gazeux, pour le calcul du pire cas."""

TEMPERATURE_MINIMALE = 273.15
"""Borne basse plausible : de l'eau liquide, donc au moins zero Celsius."""

_CENT = 100
_DEPART = (sp.Symbol("x_0", positive=True), sp.Symbol("y_0", positive=True))


def _macros_air() -> dict[str, str]:
    reservoir = Pression(
        pascals=enonce.MANOMETRE.pression_amont(
            aval=enonce.PRESSION_ATMOSPHERIQUE,
        ).pascals
    )
    rho = AIR.masse_volumique(reservoir, TEMPERATURE_AMBIANTE)
    part = part_negligee(AIR, reservoir, CIEL_GAZEUX, TEMPERATURE_AMBIANTE)
    return {
        "MasseMolaireAir": nombre(AIR.masse_molaire * 1_000, 0),
        "TemperatureAmbiante": nombre(TEMPERATURE_AMBIANTE, 2),
        "MasseVolumiqueAir": nombre(rho.kg_par_m3, 2),
        "RapportEauAir": nombre(enonce.EAU.kg_par_m3 / rho.kg_par_m3, 0),
        "HauteurCielGazeux": nombre(CIEL_GAZEUX.metres, 1),
        "PartAirNegligeePourcent": f"{part * _CENT:.4f}",
        "ImprecisionLecturePourcent": _imprecision_de_lecture(reservoir),
        **_macros_robustesse(),
    }


def _macros_robustesse() -> dict[str, str]:
    """De quoi montrer que (H3) ne repose sur aucun choix arbitraire."""
    majorant = majorant_de_la_part_negligee(
        AIR,
        CIEL_GAZEUX_MAJORANT,
        TEMPERATURE_MINIMALE,
    )
    return {
        "PartNegligeeLitterale": expression_latex(part_negligee_litterale()),
        "CielGazeuxMajorant": nombre(CIEL_GAZEUX_MAJORANT.metres, 0),
        "TemperatureMinimale": nombre(TEMPERATURE_MINIMALE, 2),
        "MajorantPartNegligeePourcent": f"{majorant * _CENT:.3f}",
    }


_MILLIMETRE = Hauteur(metres=0.001)


def _imprecision_de_lecture(reservoir: Pression) -> str:
    """Part de p1 qu'un millimetre d'erreur sur le mercure fait varier.

    C'est l'etalon auquel comparer toute approximation : inutile de
    raffiner un terme plus petit que l'incertitude de lecture.
    """
    erreur = variation_de_pression(enonce.MERCURE, _MILLIMETRE).pascals
    return f"{_CENT * erreur / reservoir.pascals:.2f}"


def _macros_emission() -> dict[str, str]:
    champ = enonce.CHAMP_C
    abscisse, ordonnee = ligne_emission(champ.u, champ.v, *_DEPART)
    invariant = invariant_de_la_ligne_emission(champ.u, champ.v, *_DEPART)
    return {
        "EmissionCX": expression_latex(abscisse),
        "EmissionCY": expression_latex(ordonnee),
        "InvariantEmissionC": expression_latex(invariant),
    }


def macros_hypotheses() -> dict[str, str]:
    """Rend les justifications chiffrees citees par le corrige."""
    return {**_macros_air(), **_macros_emission()}
