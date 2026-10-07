"""Macros de l'exercice cote du 17 mai 2021 (Mons). Prefixe `CoteMonsVingtEtUn`."""

from __future__ import annotations

import numpy as np

from polytech.electromagnetisme.application import cote_2021_mons_capacite as cp
from polytech.electromagnetisme.application import cote_2021_mons_rails as ra
from polytech.electromagnetisme.application import cote_2021_mons_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)

_MICROSECONDES = np.linspace(0.0, 40.0, 81)
"""L'abscisse de la courbe de la question 3."""


def _rails() -> dict[str, str]:
    selon_x, _, _ = ra.force()
    return {
        "CoteMonsVingtEtUnFem": nombre(ra.fem(), 2),
        "CoteMonsVingtEtUnCourant": nombre(ra.courant(), 3),
        "CoteMonsVingtEtUnPuissance": nombre(ra.puissance(), 3),
        "CoteMonsVingtEtUnForce": nombre(selon_x, 4),
    }


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    return {
        "CoteMonsVingtEtUnResistance": expression_latex(resistance),
        "CoteMonsVingtEtUnCourtCircuit": expression_latex(tension / resistance),
        "CoteMonsVingtEtUnDipole": expression_latex(th.courant_par_l_equivalent()),
        "CoteMonsVingtEtUnDipoleDecimal": nombre(
            float(th.courant_par_l_equivalent()) * 1e3, 0
        ),
    }


def _capacite() -> dict[str, str]:
    volts = np.array([cp.tension(float(t) * 1e-6) for t in _MICROSECONDES])
    points = " ".join(
        f"({nombre(float(t), 1)},{nombre(float(u), 3)})"
        for t, u in zip(_MICROSECONDES, volts, strict=True)
    )
    energie = 0.5 * float(cp.C) * float(cp.tension_au_basculement()) ** 2
    return {
        "CoteMonsVingtEtUnCourbe": points,
        "CoteMonsVingtEtUnBascule": nombre(float(cp.tension_au_basculement()), 2),
        "CoteMonsVingtEtUnMiCharge": nombre(cp.tension(8e-6), 2),
        "CoteMonsVingtEtUnHuitApres": nombre(cp.tension(24e-6), 2),
        "CoteMonsVingtEtUnSeizeApres": nombre(cp.tension(32e-6), 2),
        "CoteMonsVingtEtUnEnergie": nombre(energie * 1e6, 1),
    }


def macros_cote_mons_vingt_et_un() -> dict[str, str]:
    """Les macros des trois questions."""
    return {**_rails(), **_thevenin(), **_capacite()}
