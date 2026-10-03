"""Macros de la seance 3.2, exercices 2 et 7 : les litteraux.

Exercice 2 : la tension du condensateur au basculement vient du regime
sinusoidal de la position 1, lu a l'instant zero ; la transformee du
courant vient du resolveur, condition initiale comprise. Ses deux bouts
-- valeur initiale et valeur finale -- sont publies comme controles.

Exercice 7 : le regime force, la constante qui le raccorde a `i(0) = 0`,
et la constante de temps. La reponse complete est leur somme, et c'est
sous cette forme qu'elle se lit.

Les prefixes sont `TroisDeuxDeux` et `TroisDeuxSept`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import symboles_seance_32 as d
from polytech.electromagnetisme.application.enonces_seance_32_transitoires import (
    courant_du_deux,
    regime_du_sept,
    tension_initiale_du_deux,
)
from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    valeur_finale,
    valeur_initiale,
)


def _macros_du_deux() -> dict[str, str]:
    """La tension au basculement, la transformee, ses deux bouts."""
    courant = courant_du_deux()
    ordre = sp.degree(sp.denom(sp.together(courant)), VARIABLE)
    return {
        "TroisDeuxDeuxTensionInitiale": expression_latex(tension_initiale_du_deux()),
        "TroisDeuxDeuxTransformee": expression_latex(courant),
        "TroisDeuxDeuxCourantInitial": expression_latex(
            sp.simplify(valeur_initiale(courant)),
        ),
        "TroisDeuxDeuxCourantFinal": expression_latex(
            sp.simplify(valeur_finale(courant)),
        ),
        "TroisDeuxDeuxOrdre": str(ordre),
    }


def _macros_du_sept() -> dict[str, str]:
    """Le regime, la constante de raccord, la constante de temps."""
    regime = regime_du_sept()
    return {
        "TroisDeuxSeptRegime": expression_latex(regime),
        "TroisDeuxSeptConstante": expression_latex(sp.simplify(-regime.subs(TEMPS, 0))),
        "TroisDeuxSeptTau": expression_latex(d.L / d.R),
        "TroisDeuxSeptOmegaTau": str(d.ILLUSTRATION_OMEGA_TAU),
    }


def macros_des_litteraux() -> dict[str, str]:
    """Les deux exercices litteraux de la seance."""
    return {**_macros_du_deux(), **_macros_du_sept()}
