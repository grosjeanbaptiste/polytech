"""Macros de l'exercice 7 de statique : le meme probleme que le 5.

L'exercice 7 reprend la charge devant un plan conducteur, par la route de
Gauss et une projection trigonometrique. Il aboutit a la **meme**
densite induite que l'exercice 5, aux lettres pres.

Ce jeu de macros publie donc les deux cotes : ce que la route de Gauss
donne, et ce que le module de l'exercice 5 donne sous le changement de
lettres `(a, y, z) -> (h, x, 0)`. Que les deux macros portent la meme
expression est l'apport du corrige, et le document les met en regard
plutot que de l'affirmer.

Entierement litteral, comme les exercices 5 de statique et 5 du
transitoire : le jeu publie des expressions LaTeX, non des valeurs.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.electrostatique.gauss import (
    champ_d_une_charge,
    champ_par_projection,
    charge_dans_une_boule_uniforme,
    distance_au_point,
    divergence_du_champ_radial,
)
from polytech.electromagnetisme.domain.electrostatique.images import (
    PERMITTIVITE,
    charge_induite_totale,
    densite_induite,
    force_sur_la_charge,
)

CHARGE = sp.Symbol("Q", positive=True)
HAUTEUR = sp.Symbol("h", positive=True)
ABSCISSE = sp.Symbol("x", real=True)
RAYON = sp.Symbol("r", positive=True)

EXPOSANT_COULOMBIEN = 2
"""L'exposant du champ radial, seul a rendre la divergence nulle."""

EXPOSANT_FAUTIF = 3
"""Un exposant voisin, publie pour montrer que la divergence le refuse."""


def _macros_de_gauss() -> dict[str, str]:
    """Le champ d'une charge, son flux, et la divergence qui fixe l'exposant."""
    champ = champ_d_une_charge(CHARGE, RAYON)
    return {
        "StatiqueSeptChampDUneCharge": expression_latex(champ),
        "StatiqueSeptChargeEnlacee": expression_latex(
            charge_dans_une_boule_uniforme(
                CHARGE, sp.Symbol("R", positive=True), RAYON
            ),
        ),
        "StatiqueSeptFlux": expression_latex(
            sp.simplify(champ * 4 * sp.pi * RAYON**2),
        ),
        "StatiqueSeptDivergence": expression_latex(
            divergence_du_champ_radial(EXPOSANT_COULOMBIEN),
        ),
        "StatiqueSeptDivergenceFautive": expression_latex(
            sp.simplify(divergence_du_champ_radial(EXPOSANT_FAUTIF)),
        ),
    }


def _macros_de_la_projection() -> dict[str, str]:
    """La geometrie du manuscrit : la distance et les deux projections."""
    distance = distance_au_point(HAUTEUR, ABSCISSE)
    return {
        "StatiqueSeptDistance": expression_latex(distance),
        "StatiqueSeptSinus": expression_latex(HAUTEUR / distance),
        "StatiqueSeptCosinus": expression_latex(ABSCISSE / distance),
        "StatiqueSeptChamp": expression_latex(
            sp.simplify(champ_par_projection(CHARGE, HAUTEUR, ABSCISSE)),
        ),
    }


def _macros_de_l_identite() -> dict[str, str]:
    """La densite par les deux routes, et les ajouts au corrige officiel."""
    par_le_cinq = densite_induite(CHARGE, HAUTEUR, ABSCISSE, sp.Integer(0))
    au_pied = densite_induite(CHARGE, HAUTEUR, sp.Integer(0), sp.Integer(0))
    return {
        "StatiqueSeptDensite": expression_latex(
            sp.simplify(PERMITTIVITE * champ_par_projection(CHARGE, HAUTEUR, ABSCISSE)),
        ),
        "StatiqueSeptDensiteParLeCinq": expression_latex(sp.simplify(par_le_cinq)),
        "StatiqueSeptDensiteAuPied": expression_latex(sp.simplify(au_pied)),
        "StatiqueSeptChargeInduite": expression_latex(
            charge_induite_totale(CHARGE, HAUTEUR),
        ),
        "StatiqueSeptForce": expression_latex(force_sur_la_charge(CHARGE, HAUTEUR)),
        "StatiqueSeptPermittivite": expression_latex(PERMITTIVITE),
    }


def macros_statique_sept() -> dict[str, str]:
    """Gauss, la projection, et l'identite avec l'exercice 5."""
    return {
        **_macros_de_gauss(),
        **_macros_de_la_projection(),
        **_macros_de_l_identite(),
    }
