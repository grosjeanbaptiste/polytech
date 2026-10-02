"""Macros de la seance 2.1, exercice 2 : le fil et la spire rectangulaire.

Entierement litteral. Le jeu publie chaque resultat par les **deux**
routes quand il y en a deux -- la force cote par cote et par les travaux
virtuels, la f.e.m. par le flux et par le champ electromoteur -- au
moyen de fonctions distinctes du domaine. Que les macros portent la meme
expression est ce que le document montre.

Les logarithmes passent par `forme_logarithmique`, avec le rapport que
le manuscrit emploie : `d / (a + d)`.

Le prefixe est `DeuxUnDeux`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    forme_logarithmique,
)
from polytech.electromagnetisme.domain.magnetostatique.fem_de_la_spire import (
    fem_a_courant_sinusoidal,
    fem_de_translation,
    fem_par_le_champ_electromoteur,
)
from polytech.electromagnetisme.domain.magnetostatique.fil_et_spire import (
    SpireDevantUnFil,
    force_cote_par_cote,
    force_par_les_travaux_virtuels,
    induction_du_fil,
    resultante,
)

PERMEABILITE = sp.Symbol("mu_0", positive=True)
COURANT_DU_FIL = sp.Symbol("I_1", positive=True)
COURANT_DE_LA_SPIRE = sp.Symbol("I_2", positive=True)
EFFICACE = sp.Symbol("I", positive=True)
LARGEUR = sp.Symbol("a", positive=True)
HAUTEUR = sp.Symbol("b", positive=True)
DISTANCE = sp.Symbol("d", positive=True)
VITESSE = sp.Symbol("v", positive=True)
ANGLE = sp.Symbol("alpha", positive=True)

SPIRE = SpireDevantUnFil(largeur=LARGEUR, hauteur=HAUTEUR, distance=DISTANCE)
RAPPORT = DISTANCE / (LARGEUR + DISTANCE)
"""Le rapport sous le logarithme, dans l'ecriture du manuscrit."""


def _macros_de_la_mutuelle() -> dict[str, str]:
    """L'induction du fil, le flux, et l'inductance mutuelle."""
    abscisse = sp.Symbol("x", positive=True)
    return {
        "DeuxUnDeuxInduction": expression_latex(
            induction_du_fil(COURANT_DU_FIL, abscisse, PERMEABILITE),
        ),
        "DeuxUnDeuxFlux": forme_logarithmique(
            SPIRE.flux(COURANT_DU_FIL, PERMEABILITE),
            RAPPORT,
        ),
        "DeuxUnDeuxMutuelle": forme_logarithmique(
            SPIRE.inductance_mutuelle(PERMEABILITE),
            RAPPORT,
        ),
    }


def _macros_de_la_force() -> dict[str, str]:
    """Les quatre cotes, leur resultante, et la route des travaux virtuels."""
    forces = force_cote_par_cote(
        SPIRE, COURANT_DU_FIL, COURANT_DE_LA_SPIRE, PERMEABILITE
    )
    virtuelle = force_par_les_travaux_virtuels(
        SPIRE,
        COURANT_DU_FIL,
        COURANT_DE_LA_SPIRE,
        PERMEABILITE,
    )
    moment = COURANT_DE_LA_SPIRE * LARGEUR * HAUTEUR
    gradient = PERMEABILITE * COURANT_DU_FIL / (2 * sp.pi * DISTANCE**2)
    return {
        "DeuxUnDeuxForceGauche": expression_latex(sp.simplify(forces["gauche"][0])),
        "DeuxUnDeuxForceDroite": expression_latex(sp.simplify(forces["droit"][0])),
        "DeuxUnDeuxForceHaut": forme_logarithmique(forces["haut"][1], 1 / RAPPORT),
        "DeuxUnDeuxForceBas": forme_logarithmique(forces["bas"][1], 1 / RAPPORT),
        "DeuxUnDeuxSommeHautBas": expression_latex(resultante(forces)[1]),
        "DeuxUnDeuxForce": expression_latex(sp.factor(resultante(forces)[0])),
        "DeuxUnDeuxForceVirtuelle": expression_latex(sp.factor(virtuelle[0])),
        "DeuxUnDeuxForceAuLoin": expression_latex(moment * gradient),
    }


def _macros_des_fem() -> dict[str, str]:
    """Translation -- par deux routes --, translation oblique, courant variable."""
    oblique = VITESSE * sp.sin(ANGLE)
    arguments = (SPIRE, COURANT_DU_FIL, VITESSE, PERMEABILITE)
    return {
        "DeuxUnDeuxFem": expression_latex(sp.factor(fem_de_translation(*arguments))),
        "DeuxUnDeuxFemParCirculation": expression_latex(
            sp.factor(fem_par_le_champ_electromoteur(*arguments)),
        ),
        "DeuxUnDeuxFemOblique": expression_latex(
            sp.factor(fem_de_translation(SPIRE, COURANT_DU_FIL, oblique, PERMEABILITE)),
        ),
        "DeuxUnDeuxFemSinusoidale": forme_logarithmique(
            fem_a_courant_sinusoidal(SPIRE, EFFICACE, PERMEABILITE),
            RAPPORT,
        ),
    }


def macros_de_la_spire() -> dict[str, str]:
    """La mutuelle, la force par deux routes, et les f.e.m. induites."""
    return {
        **_macros_de_la_mutuelle(),
        **_macros_de_la_force(),
        **_macros_des_fem(),
    }
