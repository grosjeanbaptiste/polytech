"""Macros de l'exercice 8 : une sphere a charge radiale en racine du rayon.

Le corrige manuscrit est, pour une fois, complet : il mene l'exercice par
les **deux** methodes -- Poisson puis Gauss -- et constate qu'elles
s'accordent.

Ce jeu publie donc les deux resultats separement, par les deux fonctions
distinctes du domaine. Que les macros portent la meme expression est le
constat du manuscrit, et le document les met en regard plutot que de
l'affirmer.

S'y ajoutent les quatre choses que le manuscrit ne dit pas : le terme
divergent qu'il ecarte sans argument, la charge totale, l'identite du
champ exterieur avec celui d'une charge ponctuelle, et le fait que le
champ soit maximal sur la surface.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.electrostatique.sphere_radiale import (
    PERMITTIVITE,
    champ_exterieur,
    champ_par_gauss,
    champ_par_poisson,
    charge_contenue,
    charge_volumique,
    divergence_spherique,
    terme_divergent_ecarte,
)

CONSTANTE = sp.Symbol("A", positive=True)
RAYON = sp.Symbol("r", positive=True)
RAYON_DE_LA_SPHERE = sp.Symbol("R", positive=True)
CONSTANTE_D_INTEGRATION = sp.Symbol("C", positive=True)


def _macros_de_l_enonce() -> dict[str, str]:
    """La densite, et la divergence qui redonne l'equation de Poisson."""
    champ = champ_par_poisson(CONSTANTE, RAYON)
    return {
        "StatiqueHuitDensite": expression_latex(charge_volumique(CONSTANTE, RAYON)),
        "StatiqueHuitPermittivite": expression_latex(PERMITTIVITE),
        "StatiqueHuitDivergenceDuChamp": expression_latex(
            sp.simplify(divergence_spherique(champ, RAYON)),
        ),
        "StatiqueHuitSecondMembre": expression_latex(
            charge_volumique(CONSTANTE, RAYON) / PERMITTIVITE,
        ),
    }


def _macros_des_deux_routes() -> dict[str, str]:
    """Le champ interieur par Poisson, puis par Gauss."""
    return {
        "StatiqueHuitChampParPoisson": expression_latex(
            champ_par_poisson(CONSTANTE, RAYON),
        ),
        "StatiqueHuitChampParGauss": expression_latex(
            champ_par_gauss(CONSTANTE, RAYON),
        ),
        "StatiqueHuitTermeEcarte": expression_latex(
            terme_divergent_ecarte(CONSTANTE_D_INTEGRATION, RAYON),
        ),
        "StatiqueHuitDivergenceDuTermeEcarte": expression_latex(
            sp.simplify(
                divergence_spherique(
                    terme_divergent_ecarte(CONSTANTE_D_INTEGRATION, RAYON),
                    RAYON,
                ),
            ),
        ),
    }


def _macros_de_l_exterieur() -> dict[str, str]:
    """La charge totale, le champ au-dela, et la valeur a la surface."""
    totale = charge_contenue(CONSTANTE, RAYON_DE_LA_SPHERE)
    dehors = champ_exterieur(CONSTANTE, RAYON, RAYON_DE_LA_SPHERE)
    return {
        "StatiqueHuitChargeContenue": expression_latex(
            charge_contenue(CONSTANTE, RAYON),
        ),
        "StatiqueHuitChargeTotale": expression_latex(totale),
        "StatiqueHuitChampExterieur": expression_latex(dehors),
        "StatiqueHuitChampALaSurface": expression_latex(
            sp.simplify(champ_par_poisson(CONSTANTE, RAYON_DE_LA_SPHERE)),
        ),
        "StatiqueHuitPenteInterieure": expression_latex(
            sp.simplify(sp.diff(champ_par_poisson(CONSTANTE, RAYON), RAYON)),
        ),
        "StatiqueHuitPenteExterieure": expression_latex(
            sp.simplify(sp.diff(dehors, RAYON)),
        ),
    }


def macros_statique_huit() -> dict[str, str]:
    """Les deux routes, le terme ecarte, et l'exterieur."""
    return {
        **_macros_de_l_enonce(),
        **_macros_des_deux_routes(),
        **_macros_de_l_exterieur(),
    }
