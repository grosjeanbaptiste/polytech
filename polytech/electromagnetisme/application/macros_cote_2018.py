"""Macros de l'exercice cote du 16 mai 2018. Prefixe `CoteDixHuit`.

Les questions 1 et 3 publient des nombres ; les questions 2, 4 et 5,
posees en lettres, publient des expressions.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import cote_2018_bobines as bo
from polytech.electromagnetisme.application import cote_2018_commutation as cm
from polytech.electromagnetisme.application import cote_2018_fils as fi
from polytech.electromagnetisme.application import cote_2018_thevenin as th
from polytech.electromagnetisme.application import cote_2018_tore as to
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
    scientifique,
)


def _tore() -> dict[str, str]:
    hipersil, toles, air = to.partage()
    return {
        "CoteDixHuitChampDEntrefer": scientifique(to.champ_d_entrefer(), 4),
        "CoteDixHuitLongueur": nombre(to.longueur_d_un_demi_tore() * 1e2, 2),
        "CoteDixHuitHipersil": nombre(hipersil, 1),
        "CoteDixHuitToles": nombre(toles, 1),
        "CoteDixHuitAir": nombre(air, 1),
        "CoteDixHuitPartDeLAir": nombre(to.circuit().part_des_entrefers() * 100, 1),
        "CoteDixHuitEpaisseur": nombre(to.epaisseur() * 1e3, 3),
    }


def _fils() -> dict[str, str]:
    _, vertical = fi.champ()
    return {
        "CoteDixHuitSeuil": expression_latex(fi.seuil_d_existence()),
        "CoteDixHuitChampALOrigine": expression_latex(
            sp.together(vertical.subs(fi.Y, 0))
        ),
    }


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    de_la_tension, du_courant = th.tension_par_superposition()
    return {
        "CoteDixHuitTension": nombre(float(tension), 2),
        "CoteDixHuitResistance": nombre(float(resistance), 2),
        "CoteDixHuitPartDeE": nombre(float(de_la_tension), 2),
        "CoteDixHuitPartDeI": nombre(float(du_courant), 2),
        "CoteDixHuitCourant": nombre(float(th.courant_de_charge()), 2),
    }


def _commutation_et_bobines() -> dict[str, str]:
    _, pente = cm.depart_du_courant()
    return {
        "CoteDixHuitImage": expression_latex(cm.image_du_courant()),
        "CoteDixHuitPente": expression_latex(pente),
        "CoteDixHuitRapport": expression_latex(bo.rapport_des_tensions()),
        "CoteDixHuitDephasage": expression_latex(bo.dephasage()),
    }


def macros_cote_dix_huit() -> dict[str, str]:
    """Les macros des cinq questions."""
    return {**_tore(), **_fils(), **_thevenin(), **_commutation_et_bobines()}
