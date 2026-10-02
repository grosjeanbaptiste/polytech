"""Macros de l'exercice « Statique 1.1 » : le cylindre charge.

Entierement litteral. Les deux routes du champ -- Poisson et Gauss --
sont publiees separement, par les deux fonctions distinctes du domaine :
leur egalite est l'apport du corrige, et le document les met en regard
plutot que de l'affirmer.

Le prefixe est `StatiqueUnUn` : une macro LaTeX ne porte pas de chiffre,
et le numero « 1.1 » de l'enonce s'ecrit donc en lettres.

Deux macros publient un **rayon different** pour le meme champ. C'est
voulu : le rayon du cylindre se simplifie, et composer les deux cote a
cote montre l'effacement au lieu de l'annoncer.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.electrostatique.cylindre import (
    PERMITTIVITE,
    champ_par_gauss,
    champ_par_poisson,
    densite_lineique,
    densite_surfacique,
    divergence_cylindrique,
    potentiel,
)

CHARGE = sp.Symbol("Q", positive=True)
LONGUEUR = sp.Symbol("L", positive=True)
RAYON = sp.Symbol("R", positive=True)
AUTRE_RAYON = sp.Symbol("R'", positive=True)
DISTANCE = sp.Symbol("rho", positive=True)
REFERENCE = sp.Symbol("rho_0", positive=True)


def _macros_des_densites() -> dict[str, str]:
    """La densite lineique, la surfacique, et l'aire qui la divise."""
    return {
        "StatiqueUnUnPermittivite": expression_latex(PERMITTIVITE),
        "StatiqueUnUnLineique": expression_latex(
            densite_lineique(CHARGE, LONGUEUR),
        ),
        "StatiqueUnUnSurfacique": expression_latex(
            densite_surfacique(CHARGE, LONGUEUR, RAYON),
        ),
        "StatiqueUnUnAireLaterale": expression_latex(
            2 * sp.pi * RAYON * LONGUEUR,
        ),
    }


def _macros_des_deux_routes() -> dict[str, str]:
    """Le champ par Poisson, par Gauss, et avec un autre rayon."""
    return {
        "StatiqueUnUnChampParPoisson": expression_latex(
            champ_par_poisson(CHARGE, LONGUEUR, RAYON, DISTANCE),
        ),
        "StatiqueUnUnChampParGauss": expression_latex(
            champ_par_gauss(CHARGE, LONGUEUR, DISTANCE),
        ),
        "StatiqueUnUnChampAutreRayon": expression_latex(
            champ_par_poisson(CHARGE, LONGUEUR, AUTRE_RAYON, DISTANCE),
        ),
        "StatiqueUnUnAutreRayon": expression_latex(AUTRE_RAYON),
    }


def _macros_des_controles() -> dict[str, str]:
    """Les deux bouclages : la divergence nulle et le champ a la surface."""
    champ = champ_par_poisson(CHARGE, LONGUEUR, RAYON, DISTANCE)
    en_carre = CHARGE / DISTANCE**2
    return {
        "StatiqueUnUnDivergence": expression_latex(
            sp.simplify(divergence_cylindrique(champ, DISTANCE)),
        ),
        "StatiqueUnUnDivergenceEnCarre": expression_latex(
            sp.simplify(divergence_cylindrique(en_carre, DISTANCE)),
        ),
        "StatiqueUnUnChampALaSurface": expression_latex(
            sp.simplify(champ_par_poisson(CHARGE, LONGUEUR, RAYON, RAYON)),
        ),
        "StatiqueUnUnRelationDePassage": expression_latex(
            sp.simplify(
                PERMITTIVITE * champ_par_poisson(CHARGE, LONGUEUR, RAYON, RAYON),
            ),
        ),
    }


def _macros_du_potentiel() -> dict[str, str]:
    """Le potentiel logarithmique, et le gradient qui redonne le champ."""
    tension = potentiel(CHARGE, LONGUEUR, DISTANCE, REFERENCE)
    return {
        "StatiqueUnUnPotentiel": expression_latex(tension),
        "StatiqueUnUnGradientDuPotentiel": expression_latex(
            sp.simplify(-sp.diff(tension, DISTANCE)),
        ),
    }


def macros_statique_un_un() -> dict[str, str]:
    """Les densites, les deux routes, les controles et le potentiel."""
    return {
        **_macros_des_densites(),
        **_macros_des_deux_routes(),
        **_macros_des_controles(),
        **_macros_du_potentiel(),
    }
