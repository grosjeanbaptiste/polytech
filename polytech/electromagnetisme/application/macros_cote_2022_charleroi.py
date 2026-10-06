"""Macros de l'exercice cote du 29 avril 2022 (Charleroi). Prefixe `CoteVingtDeux`.

La question 1 publie des expressions, la question 2 des nombres ; la
question 3, litterale, n'a besoin d'aucune macro.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import cote_2022_coquille as cq
from polytech.electromagnetisme.application import cote_2022_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _coquille() -> dict[str, str]:
    _, coquille, dehors = cq.champs()
    interieure, exterieure = cq.charges_surfaciques()
    saut = sp.simplify(dehors.subs(cq.RAYON, cq.R2) / coquille.subs(cq.RAYON, cq.R2))
    return {
        "CoteVingtDeuxSaut": expression_latex(saut),
        "CoteVingtDeuxFaceInterieure": expression_latex(interieure),
        "CoteVingtDeuxFaceExterieure": expression_latex(exterieure),
    }


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    aux_bornes, energie = th.charge_du_dipole()
    return {
        "CoteVingtDeuxTension": expression_latex(tension),
        "CoteVingtDeuxResistance": expression_latex(resistance),
        "CoteVingtDeuxCourtCircuit": expression_latex(th.courant_de_court_circuit()),
        "CoteVingtDeuxTensionDeCZero": expression_latex(aux_bornes),
        "CoteVingtDeuxTensionDecimale": nombre(float(aux_bornes), 2),
        "CoteVingtDeuxEnergie": nombre(float(energie) * 1e6, 2),
    }


def macros_cote_vingt_deux() -> dict[str, str]:
    """Les macros des trois questions."""
    return {**_coquille(), **_thevenin()}
