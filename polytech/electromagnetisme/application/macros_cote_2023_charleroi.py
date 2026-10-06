"""Macros de l'exercice cote du 17 avril 2023 (Charleroi). Prefixe `CoteVingtTrois`.

La question 1 publie des nombres ; les questions 2 et 3, litterales,
publient des expressions.
"""

from __future__ import annotations

from polytech.electromagnetisme.application import cote_2023_demi_tores as dt
from polytech.electromagnetisme.application import cote_2023_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    courant = th.courant_dans_r2()
    return {
        "CoteVingtTroisTension": expression_latex(tension),
        "CoteVingtTroisTensionDecimale": nombre(float(tension), 2),
        "CoteVingtTroisResistance": expression_latex(resistance),
        "CoteVingtTroisResistanceDecimale": nombre(float(resistance), 2),
        "CoteVingtTroisCourant": expression_latex(courant),
        "CoteVingtTroisCourantDecimal": nombre(float(courant), 2),
        "CoteVingtTroisCourtCircuit": expression_latex(th.courant_de_court_circuit()),
    }


def _demi_tores() -> dict[str, str]:
    return {
        "CoteVingtTroisFlux": expression_latex(dt.flux()),
        "CoteVingtTroisFem": expression_latex(dt.fem()),
        "CoteVingtTroisFemInitiale": expression_latex(dt.fem().subs(dt.T, 0)),
        "CoteVingtTroisMutuelle": expression_latex(dt.inductance_mutuelle()),
        "CoteVingtTroisAire": expression_latex(dt.aire_sous_la_fem()),
    }


def macros_cote_vingt_trois() -> dict[str, str]:
    """Les macros des questions 1 et 2 ; la question 3 n'en a pas besoin."""
    return {**_thevenin(), **_demi_tores()}
