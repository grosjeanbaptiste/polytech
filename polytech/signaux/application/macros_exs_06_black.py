"""Macros de la feuille « Analyse harmonique », exercices 3, 5 et 6.

Prefixe `ExsSix`.
"""

from __future__ import annotations

from polytech.signaux.application import black_exs_06 as bl
from polytech.signaux.application import bode_exs_06 as bo
from polytech.signaux.application import courbes_exs_06 as co
from polytech.signaux.application import second_ordre_exs_06 as so
from polytech.signaux.application.formatage import nombre
from polytech.signaux.domain.trace_asymptotique import trace_d_une_transmittance


def _exercice_trois() -> dict[str, str]:
    pulsation, argument = bl.passage_a_zero_decibel()
    return {
        "ExsSixTroisBlack": co.black(bl.TROIS, 0.2, 40.0),
        "ExsSixTroisPulsation": nombre(pulsation, 2),
        "ExsSixTroisArgument": nombre(argument, 1),
        "ExsSixTroisPassage": co.point(argument, 0.0),
        "ExsSixTroisMarge": nombre(bl.marges().de_phase, 1),
    }


def _exercice_cinq() -> dict[str, str]:
    macros = {}
    for nom, lettre in (("T1", "Un"), ("T2", "Deux"), ("T3", "Trois")):
        macros[f"ExsSixCinqBlack{lettre}"] = co.black(bl.FONCTIONS[nom], 0.003, 30.0)
        macros[f"ExsSixCinqExtreme{lettre}"] = nombre(bl.extreme_de_l_argument(nom), 1)
    return macros


def _exercice_six() -> dict[str, str]:
    pulsation, surtension = so.resonance()
    propre, _ = so.point_propre()
    trace = trace_d_une_transmittance(so.TRANSMITTANCE)
    _, _, amortissement = so.parametres()
    return {
        "ExsSixSixAsymptote": co.asymptote(trace, 0.1, 10.0),
        "ExsSixSixGain": co.gain(so.TRANSMITTANCE, 0.1, 10.0),
        "ExsSixSixArgument": co.argument(so.TRANSMITTANCE, 0.1, 10.0),
        "ExsSixSixAmortissement": nombre(float(amortissement), 1),
        "ExsSixSixResonance": nombre(pulsation, 3),
        "ExsSixSixSurtension": nombre(surtension, 2),
        "ExsSixSixSommet": nombre(bo.gain_reel(so.TRANSMITTANCE, pulsation), 2),
        "ExsSixSixGainPropre": nombre(propre, 2),
    }


def macros_de_black() -> dict[str, str]:
    """Les macros des exercices 3, 5 et 6."""
    return {**_exercice_trois(), **_exercice_cinq(), **_exercice_six()}
