"""Macros de la partie I : statique des fluides.

Chaque entree est le rendu LaTeX d'une expression *derivee* par le domaine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.fluides.application import enonce
from polytech.fluides.application.formatage import expression_latex
from polytech.fluides.domain.statique.parements import (
    ParementIncline,
    ParementTriangulaire,
    ParementVertical,
)
from polytech.fluides.domain.statique.symboles import H

if TYPE_CHECKING:
    from polytech.fluides.domain.statique.efforts import Parement

_MOITIE = sp.Rational(1, 2)
_CENT = 100


def _macros_bras_de_levier(mur: Parement, vallee: Parement) -> dict[str, str]:
    """Le bras se deduit de la ligne d'action, et differe selon la geometrie."""
    return {
        "EquationLigneAction": expression_latex(mur.equation_ligne_action()),
        "BrasLevierVertical": expression_latex(mur.expression_bras_de_levier()),
        "BrasLevierValleeV": expression_latex(vallee.expression_bras_de_levier()),
        "MomentValleeV": expression_latex(vallee.expression_moment()),
        # Bras de levier une fois la hauteur reduite : (H/2)/3 = H/6. Le
        # corrige l'ecrivait a la main, y compris dans une figure.
        "BrasLevierHalve": expression_latex(
            mur.expression_bras_de_levier().subs(
                H,
                mur.hauteur_pour_fraction_de_volume(_MOITIE) * H,
            ),
        ),
    }


def _macros_facteurs(mur: Parement, vallee: Parement) -> dict[str, str]:
    """Le corrige dit « divisee par 4 » : le 4 vient du rapport, inverse."""
    facteurs = {
        "FacteurPousseePrismatique": mur.rapport_poussee(_MOITIE),
        "FacteurMomentPrismatique": mur.rapport_moment(_MOITIE),
        "FacteurPousseeValleeV": vallee.rapport_poussee(_MOITIE),
    }
    perte = _CENT * (1 - float(vallee.hauteur_pour_fraction_de_volume(_MOITIE)))
    return {
        **{nom: expression_latex(1 / valeur) for nom, valeur in facteurs.items()},
        "PerteHauteurValleeVPourcent": f"{perte:.0f}",
    }


def macros_statique() -> dict[str, str]:
    """Rend les raisonnements litteraux de la statique."""
    mur = ParementVertical(fluide=enonce.EAU, largeur=1.0)
    vallee = ParementTriangulaire(fluide=enonce.EAU, pente=0.5)
    return {
        **_macros_bras_de_levier(mur, vallee),
        **_macros_facteurs(mur, vallee),
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
