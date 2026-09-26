"""Macros de la partie I : statique des fluides.

Chaque entree est le rendu LaTeX d'une expression *derivee* par le domaine.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex
from polytech.fluides.domain.statique.parements import (
    ParementIncline,
    ParementTriangulaire,
    ParementVertical,
)

_MOITIE = sp.Rational(1, 2)


def macros_statique() -> dict[str, str]:
    """Rend les raisonnements litteraux de la statique."""
    mur = ParementVertical(fluide=enonce.EAU, largeur=1.0)
    vallee = ParementTriangulaire(fluide=enonce.EAU, pente=0.5)
    return {
        "EquationManometre": expression_latex(enonce.MANOMETRE.equation_symbolique()),
        "PousseeVerticale": expression_latex(mur.expression_poussee()),
        "PousseeInclinee": expression_latex(
            ParementIncline(
                fluide=enonce.EAU,
                largeur=1.0,
                angle=sp.Symbol("theta", positive=True),
            ).expression_poussee(),
        ),
        "MomentVertical": expression_latex(mur.expression_moment()),
        "LigneActionVerticale": expression_latex(mur.expression_ligne_action()),
        "LigneActionValleeV": expression_latex(vallee.expression_ligne_action()),
        "PousseeValleeV": expression_latex(vallee.expression_poussee()),
        "RapportPousseePrismatique": expression_latex(mur.rapport_poussee(_MOITIE)),
        "RapportMomentPrismatique": expression_latex(mur.rapport_moment(_MOITIE)),
        "RapportPousseeValleeV": expression_latex(vallee.rapport_poussee(_MOITIE)),
        "ExposantPousseeVerticale": str(mur.exposant_poussee()),
        "ExposantPousseeValleeV": str(vallee.exposant_poussee()),
    }
