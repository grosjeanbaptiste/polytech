"""Macros de l'exercice 12 : le disque tournant dans un champ magnetique.

Entierement litteral. Le jeu publie la tension par **trois** routes -- la
force de Laplace integree le long du rayon, le flux balaye, la tige
equivalente -- par trois fonctions distinctes du domaine. Que les trois
macros portent la meme expression est ce que le document met en regard,
au lieu de l'affirmer.

S'y ajoute la derivee du flux **enlace** par un contour fixe, qui vaut
zero : c'est le paradoxe du disque, et le publier a cote de la tension
non nulle le montre.

Le prefixe est `ChampDouze`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.magnetostatique.disque_tournant import (
    courant_debite,
    tension_induite,
    tension_par_la_tige_equivalente,
    tension_par_le_flux_balaye,
    vitesse_au_rayon_moyen,
    vitesse_lineaire,
)

INDUCTION = sp.Symbol("B", positive=True)
PULSATION = sp.Symbol("omega", positive=True)
DIAMETRE = sp.Symbol("D", positive=True)
RESISTANCE = sp.Symbol("R", positive=True)
RAYON = sp.Symbol("r", positive=True)
TEMPS = sp.Symbol("t", positive=True)


def _macros_de_la_route_du_manuscrit() -> dict[str, str]:
    """La vitesse, l'integrande, la tension et le courant encadre."""
    vitesse = vitesse_lineaire(PULSATION, RAYON)
    return {
        "ChampDouzeVitesse": expression_latex(vitesse),
        "ChampDouzeIntegrande": expression_latex(INDUCTION * vitesse),
        "ChampDouzeTension": expression_latex(
            tension_induite(INDUCTION, PULSATION, DIAMETRE),
        ),
        "ChampDouzeCourant": expression_latex(
            courant_debite(INDUCTION, PULSATION, DIAMETRE, RESISTANCE),
        ),
    }


def _macros_des_deux_autres_routes() -> dict[str, str]:
    """Le flux balaye, puis la tige a la vitesse du rayon moyen."""
    par_le_flux = tension_par_le_flux_balaye(INDUCTION, PULSATION, DIAMETRE)
    return {
        "ChampDouzeAireBalayee": expression_latex(par_le_flux / INDUCTION),
        "ChampDouzeTensionParLeFlux": expression_latex(par_le_flux),
        "ChampDouzeVitesseAuBord": expression_latex(
            vitesse_lineaire(PULSATION, DIAMETRE / 2),
        ),
        "ChampDouzeVitesseMoyenne": expression_latex(
            vitesse_au_rayon_moyen(PULSATION, DIAMETRE),
        ),
        "ChampDouzeTensionParLaTige": expression_latex(
            tension_par_la_tige_equivalente(INDUCTION, PULSATION, DIAMETRE),
        ),
    }


def _macros_des_ajouts() -> dict[str, str]:
    """Le paradoxe du flux enlace, et la loi en carre du diametre."""
    simple = tension_induite(INDUCTION, PULSATION, DIAMETRE)
    double = tension_induite(INDUCTION, PULSATION, 2 * DIAMETRE)
    flux_enlace = INDUCTION * sp.pi * (DIAMETRE / 2) ** 2
    return {
        "ChampDouzeFluxEnlace": expression_latex(flux_enlace),
        "ChampDouzeDeriveeDuFluxEnlace": expression_latex(
            sp.diff(flux_enlace, TEMPS),
        ),
        "ChampDouzeRapportADoubleDiametre": expression_latex(
            sp.simplify(double / simple),
        ),
    }


def macros_champ_douze() -> dict[str, str]:
    """La route du manuscrit, les deux autres, et les deux ajouts."""
    return {
        **_macros_de_la_route_du_manuscrit(),
        **_macros_des_deux_autres_routes(),
        **_macros_des_ajouts(),
    }
