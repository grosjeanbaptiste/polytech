"""Macros de l'exercice cote du 18 mai 2026 (Mons). Prefixe `CoteMonsVingtSix`."""

from __future__ import annotations

from polytech.electromagnetisme.application import cote_2026_mons_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def macros_cote_mons_vingt_six() -> dict[str, str]:
    """Les macros de la question 1 ; les questions 2 et 3 sont litterales."""
    tension, resistance = th.thevenin()
    premier, second = th.courants_de_maille()
    return {
        "CoteMonsVingtSixResistance": expression_latex(resistance),
        "CoteMonsVingtSixResistanceDecimale": nombre(float(resistance), 2),
        "CoteMonsVingtSixTension": expression_latex(tension),
        "CoteMonsVingtSixTensionDecimale": nombre(float(tension), 2),
        "CoteMonsVingtSixCourantUn": expression_latex(premier),
        "CoteMonsVingtSixCourantDeux": expression_latex(second),
        "CoteMonsVingtSixCourtCircuit": expression_latex(th.court_circuit()),
    }
