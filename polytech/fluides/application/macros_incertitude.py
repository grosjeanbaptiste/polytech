"""Macros du budget d'incertitude du manometre.

Le corrige affichait p1 avec deux decimales et hierarchisait les sources de
travers. Ces macros publient le budget complet, seul moyen de justifier l'un
et de corriger l'autre.
"""

from __future__ import annotations

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import nombre
from polytech.fluides.domain.grandeurs import Pression
from polytech.fluides.domain.statique.incertitude import budget_du_manometre

RESOLUTION_HAUTEUR = 0.001
"""Les hauteurs sont donnees au centimetre ; on lit au millimetre."""

RESOLUTION_ATMOSPHERIQUE = Pression(pascals=100.0)
"""p_atm est donnee au dixieme de kilopascal."""

FACTEUR_ELARGISSEMENT = 2
"""k = 2, soit un intervalle de confiance d'environ 95 % (GUM)."""

_CENT = 100
_PASCALS_PAR_KILOPASCAL = 1_000.0
_NOMS = {
    "eau": "UEau",
    "huile": "UHuile",
    "Hg": "UMercure",
    "atmosphere": "UAtmosphere",
}
_PARTS = {
    "eau": "PartEauVariance",
    "huile": "PartHuileVariance",
    "Hg": "PartMercureVariance",
    "atmosphere": "PartAtmosphereVariance",
}


def macros_incertitude() -> dict[str, str]:
    """Rend le budget d'incertitude tel que le domaine le compose."""
    budget = budget_du_manometre(
        enonce.MANOMETRE,
        resolution_hauteur=RESOLUTION_HAUTEUR,
        resolution_atmospherique=RESOLUTION_ATMOSPHERIQUE,
    )
    compose = budget.ecart_type_compose()
    macros = {
        "ResolutionLectureMM": nombre(RESOLUTION_HAUTEUR * _CENT * 10, 0),
        "ResolutionAtmospheriqueKPa": nombre(
            RESOLUTION_ATMOSPHERIQUE.kilopascals,
            1,
        ),
        "IncertitudeComposeePa": nombre(compose, 1),
        "IncertitudeComposeeKPa": nombre(compose / _PASCALS_PAR_KILOPASCAL, 3),
        "IncertitudeElargieKPa": nombre(
            FACTEUR_ELARGISSEMENT * compose / _PASCALS_PAR_KILOPASCAL,
            2,
        ),
        "DecimalesJustifiees": str(budget.decimales_justifiees_en_kilopascals()),
    }
    for contribution in budget.contributions:
        macros[_NOMS[contribution.nom]] = nombre(contribution.ecart_type, 1)
        macros[_PARTS[contribution.nom] + "Pourcent"] = nombre(
            _CENT * contribution.part_de_variance(budget),
            1,
        )
    return macros
