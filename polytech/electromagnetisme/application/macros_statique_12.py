"""Macros de l'exercice 12 de statique : le travail pour ecarter deux armatures.

Entierement litteral, comme l'enonce du recueil : le jeu publie des
expressions LaTeX, non des valeurs. Chaque cas donne ses deux energies,
la force en chemin et le travail ; s'y ajoutent ce que le corrige apporte
au-dela du manuscrit -- le rapport des deux travaux, le bilan de la
source, la tension finale, l'ecartement a l'infini.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import travail_statique_12 as tr
from polytech.electromagnetisme.application.formatage import expression_latex


def avec_l_ecart(expression: sp.Expr) -> str:
    """L'expression, son facteur `b - a` ecrit dans ce sens ; leve s'il manque."""
    ecart = tr.ARRIVEE - tr.DEPART
    quotient = sp.simplify(expression / ecart)
    if sp.simplify(expression).subs(tr.ARRIVEE, tr.DEPART) != 0:
        msg = f"pas de facteur b - a dans {expression} : non nulle pour b = a"
        raise ValueError(msg)
    return rf"{expression_latex(quotient)}\,\left(b - a\right)"


def _cas(prefixe: str, cas: tr.Ecartement) -> dict[str, str]:
    """Les quatre grandeurs d'un ecartement, sous un meme prefixe."""
    return {
        f"{prefixe}Initiale": expression_latex(cas.initiale),
        f"{prefixe}Finale": expression_latex(cas.finale),
        f"{prefixe}Force": expression_latex(cas.force),
        f"{prefixe}Travail": avec_l_ecart(cas.travail),
    }


def macros_statique_douze() -> dict[str, str]:
    """Les expressions que le corrige de l'exercice 12 compose."""
    recu, perte_du_champ = tr.bilan_de_la_source()
    fini, _ = tr.a_l_infini()
    return {
        **_cas("StatiqueDouzePotentiels", tr.a_potentiels_constants()),
        **_cas("StatiqueDouzeCharges", tr.a_charges_constantes()),
        "StatiqueDouzeCapacite": expression_latex(tr.capacite()),
        "StatiqueDouzeCharge": expression_latex(tr.charge_initiale()),
        "StatiqueDouzeTravailEnCharge": avec_l_ecart(
            tr.travail_en_fonction_de_la_charge()
        ),
        "StatiqueDouzeRapport": expression_latex(tr.rapport_des_travaux()),
        "StatiqueDouzeRecuParLaSource": avec_l_ecart(recu),
        "StatiqueDouzePerteDuChamp": avec_l_ecart(perte_du_champ),
        "StatiqueDouzeTensionFinale": expression_latex(tr.tension_finale()),
        "StatiqueDouzeALInfini": expression_latex(fini),
    }
