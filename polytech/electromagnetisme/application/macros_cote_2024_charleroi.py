"""Macros de l'exercice cote du 16 avril 2024 (Charleroi). Prefixe `CoteVingtQuatre`."""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application import cote_2024_fils as fl
from polytech.electromagnetisme.application import cote_2024_phase as ph
from polytech.electromagnetisme.application import cote_2024_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    oubli_tension, oubli_resistance = th.thevenin(avec_la_bobine=False)
    return {
        "CoteVingtQuatreTension": nombre(float(tension), 1),
        "CoteVingtQuatreResistance": nombre(float(resistance), 2),
        "CoteVingtQuatreDiviseur": nombre(float(th.tension_par_le_diviseur()), 1),
        "CoteVingtQuatreEnergie": nombre(th.energie() * 1e6, 1),
        "CoteVingtQuatreOubliTension": nombre(float(oubli_tension), 1),
        "CoteVingtQuatreOubliResistance": nombre(float(oubli_resistance), 2),
    }


def _fils() -> dict[str, str]:
    h_x, h_y = fl.champ()
    selon_x, _ = fl.force_lineique(1)
    par_ampere = float(selon_x.subs(fl.COURANT, 1).subs(fl.MU0, 4e-7 * sp.pi))
    return {
        "CoteVingtQuatreChampX": expression_latex(sp.apart(h_x, fl.Y)),
        "CoteVingtQuatreChampY": expression_latex(h_y),
        "CoteVingtQuatreForce": expression_latex(selon_x),
        "CoteVingtQuatreForceParAmpere": nombre(par_ampere * 1e9, 1),
    }


def _phase() -> dict[str, str]:
    sous_condition = ph.rapport().subs(ph.L2, ph.L1 * ph.R2 / ph.R1)
    return {
        "CoteVingtQuatreRapport": expression_latex(sp.simplify(sous_condition)),
    }


def macros_cote_vingt_quatre() -> dict[str, str]:
    """Les macros des trois questions."""
    return {**_thevenin(), **_fils(), **_phase()}
