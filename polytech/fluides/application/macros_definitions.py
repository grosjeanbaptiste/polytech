"""Macros des enonces de lois du cours.

Ce ne sont pas des resultats calcules mais les definitions dont tout le
reste decoule. Les publier evite que le document en donne une version qui
diverge de celle que la bibliotheque applique reellement.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.application.formatage import expression_latex
from polytech.fluides.domain.cinematique.champ_generique import CHAMP_GENERIQUE
from polytech.fluides.domain.cinematique.lignes import definition_ligne_de_courant
from polytech.fluides.domain.statique.stevin import (
    expression_variation_de_pression,
    loi_de_stevin,
    relation_entre_deux_points,
)


def macros_definitions() -> dict[str, str]:
    """Rend les enonces de lois cites par le corrige."""
    acceleration_x, _ = CHAMP_GENERIQUE.acceleration()
    return {
        "LoiStevin": expression_latex(loi_de_stevin()),
        "VariationPression": expression_latex(expression_variation_de_pression()),
        "RelationHydrostatique": expression_latex(relation_entre_deux_points()),
        "DefinitionDivergence": expression_latex(CHAMP_GENERIQUE.divergence()),
        "DefinitionTauxRotation": expression_latex(
            sp.factor_terms(CHAMP_GENERIQUE.taux_de_rotation()),
        ),
        "DefinitionAcceleration": expression_latex(acceleration_x),
        "DefinitionLigneDeCourant": expression_latex(definition_ligne_de_courant()),
    }
