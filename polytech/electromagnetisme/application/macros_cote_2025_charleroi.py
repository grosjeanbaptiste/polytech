"""Macros de l'exercice cote du 12 mai 2025 (Charleroi). Prefixe `CoteVingtCinq`."""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import cote_2025_capteur as cp
from polytech.electromagnetisme.application import cote_2025_ouverture as ov
from polytech.electromagnetisme.application import cote_2025_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    return {
        "CoteVingtCinqTension": nombre(float(tension), 2),
        "CoteVingtCinqResistance": nombre(float(resistance), 2),
        "CoteVingtCinqMaille": nombre(float(th.courant_de_maille()), 2),
        "CoteVingtCinqCourtCircuit": nombre(float(tension / resistance), 1),
        "CoteVingtCinqDipole": nombre(float(th.tension_par_le_circuit()), 1),
    }


def _capteur() -> dict[str, str]:
    bas, haut = cp.plage_du_champ()
    return {
        "CoteVingtCinqPosition": expression_latex(cp.position()),
        "CoteVingtCinqSection": expression_latex(cp.section()),
        "CoteVingtCinqSectionCm": nombre(float(cp.section()) * 1e4, 1),
        "CoteVingtCinqChampBas": nombre(float(bas), 1),
        "CoteVingtCinqChampHaut": nombre(float(haut), 1),
        "CoteVingtCinqVitesse": nombre(float(cp.vitesse_maximale()), 2),
    }


def _ouverture() -> dict[str, str]:
    unites = dict.fromkeys((ov.E0, ov.W, ov.R0, ov.R1, ov.R2, ov.L), 1)
    depart = ov.tension().subs(unites).subs(ov.T, 0)
    return {
        "CoteVingtCinqExemple": nombre(float(depart), 3),
        "CoteVingtCinqExempleExact": expression_latex(sp.nsimplify(depart)),
    }


def macros_cote_vingt_cinq() -> dict[str, str]:
    """Les macros des trois questions."""
    return {**_thevenin(), **_capteur(), **_ouverture()}
