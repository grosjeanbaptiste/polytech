"""Macros de l'exercice cote du 16 mai 2023 (Mons). Prefixe `CoteMonsVingtTrois`."""

from __future__ import annotations

import math

from polytech.electromagnetisme.application import cote_2023_mons_bobines as bo
from polytech.electromagnetisme.application import cote_2023_mons_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    return {
        "CoteMonsVingtTroisResistance": expression_latex(resistance),
        "CoteMonsVingtTroisResistanceDecimale": nombre(float(resistance), 2),
        "CoteMonsVingtTroisTension": expression_latex(tension),
        "CoteMonsVingtTroisTensionDecimale": nombre(float(tension), 2),
        "CoteMonsVingtTroisGroupe": expression_latex(th.resistances_groupees()),
        "CoteMonsVingtTroisCourtCircuit": nombre(float(th.court_circuit()), 1),
    }


def _bobines() -> dict[str, str]:
    courant = bo.courant_du_secondaire()
    amplitude, phase = bo.sinusoide()
    fournie, dissipee, transmise = bo.puissances()
    entree = bo.impedance_d_entree()
    return {
        "CoteMonsVingtTroisReel": nombre(courant.real, 2),
        "CoteMonsVingtTroisImaginaire": nombre(-courant.imag, 2),
        "CoteMonsVingtTroisEfficace": nombre(abs(courant), 2),
        "CoteMonsVingtTroisAmplitude": nombre(amplitude, 2),
        "CoteMonsVingtTroisPhase": nombre(-phase, 2),
        "CoteMonsVingtTroisPhaseDegres": nombre(math.degrees(-phase), 1),
        "CoteMonsVingtTroisFournie": nombre(fournie, 1),
        "CoteMonsVingtTroisDissipee": nombre(dissipee, 1),
        "CoteMonsVingtTroisRendue": nombre(-transmise, 1),
        "CoteMonsVingtTroisEntree": nombre(entree.real, 2),
    }


def macros_cote_mons_vingt_trois() -> dict[str, str]:
    """Les macros des questions 1 et 3 ; la question 2 est litterale."""
    return {**_thevenin(), **_bobines()}
