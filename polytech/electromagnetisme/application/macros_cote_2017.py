"""Macros de l'exercice cote du 13 mai 2017. Prefixe `CoteDixSept`.

Les questions 1 et 2 publient des nombres ; les questions 3 et 4, que
l'enonce pose en lettres, publient des expressions.
"""

from __future__ import annotations

from polytech.electromagnetisme.application import cote_2017_fils as fi
from polytech.electromagnetisme.application import cote_2017_norton as no
from polytech.electromagnetisme.application import cote_2017_ouverture as ou
from polytech.electromagnetisme.application import cote_2017_puissance as pu
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _norton() -> dict[str, str]:
    courant, resistance = no.norton()
    dans_le_dipole = no.courant_du_dipole()
    return {
        "CoteDixSeptResistance": expression_latex(resistance),
        "CoteDixSeptTensionAVide": expression_latex(no.tension_a_vide()),
        "CoteDixSeptCourantDeNorton": expression_latex(courant),
        "CoteDixSeptCourantDeNortonDecimal": nombre(float(courant), 2),
        "CoteDixSeptResistanceEnParallele": expression_latex(
            no.resistance_en_parallele()
        ),
        "CoteDixSeptCourant": expression_latex(dans_le_dipole),
        "CoteDixSeptCourantDecimal": nombre(float(dans_le_dipole), 2),
    }


def _puissance() -> dict[str, str]:
    tension, puissance = pu.tension_de_la_branche(), pu.puissance_de_la_branche()
    bilan = pu.bilan_par_element()
    return {
        "CoteDixSeptReactance": nombre(1 / (pu.PULSATION * pu.CAPACITE), 1),
        "CoteDixSeptTensionReelle": nombre(-tension.real, 1),
        "CoteDixSeptTensionImaginaire": nombre(tension.imag, 1),
        "CoteDixSeptPuissanceActive": nombre(-puissance.real, 1),
        "CoteDixSeptPuissanceReactive": nombre(-puissance.imag, 1),
        "CoteDixSeptFournieParLaSource": nombre(-bilan.source_active, 1),
        "CoteDixSeptReactiveDeLaSource": nombre(bilan.source_reactive, 1),
        "CoteDixSeptDissipee": nombre(bilan.resistance, 1),
        "CoteDixSeptReactiveDuCondensateur": nombre(-bilan.condensateur, 1),
    }


def _ouverture() -> dict[str, str]:
    dissipee, _ = ou.energies()
    return {
        "CoteDixSeptCourantAvant": expression_latex(ou.courant_avant()),
        "CoteDixSeptCourantApres": expression_latex(ou.courant_apres()),
        "CoteDixSeptEnergie": expression_latex(dissipee),
    }


def _fils() -> dict[str, str]:
    return {
        "CoteDixSeptNappe": expression_latex(fi.charge_surfacique_du_reseau_infini()),
        "CoteDixSeptInduiteParUnFil": expression_latex(fi.charge_induite_par_un_fil()),
        "CoteDixSeptSousUnFilSeul": expression_latex(
            fi.charge_surfacique_sous_un_fil_seul()
        ),
    }


def macros_cote_dix_sept() -> dict[str, str]:
    """Les macros des quatre questions."""
    return {**_norton(), **_puissance(), **_ouverture(), **_fils()}
