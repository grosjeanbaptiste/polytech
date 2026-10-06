"""Macros de la feuille « Analyse harmonique », exercices 1, 2 et 4.

Prefixe `ExsSix`.
"""

from __future__ import annotations

from polytech.signaux.application import bode_exs_06 as bo
from polytech.signaux.application import courbes_exs_06 as co
from polytech.signaux.application.formatage import nombre

_ETENDUES = {
    "a": (0.1, 100.0),
    "b": (0.1, 100.0),
    "c": (0.01, 100.0),
    "d": (1.0, 20000.0),
    "e": (0.01, 100.0),
}
"""Les pulsations entre lesquelles chaque asymptote de l'exercice 1 est tracee."""
_NOMS = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}


def _associations() -> dict[str, str]:
    macros = {}
    for nom, etendue in _ETENDUES.items():
        macros[f"ExsSixAsymptote{_NOMS[nom]}"] = co.asymptote(bo.trace(nom), *etendue)
    macros["ExsSixPalierC"] = nombre(bo.trace("c").gain_en_decibels(0.5), 2)
    macros["ExsSixPalierB"] = nombre(bo.trace("b").gain_en_decibels(2.0), 2)
    return macros


def _exercice_deux() -> dict[str, str]:
    fonction = bo.FONCTIONS["deux"]
    pulsation, argument = bo.sommet_de_l_argument()
    return {
        "ExsSixDeuxAsymptote": co.asymptote(bo.trace("deux"), 0.1, 100.0),
        "ExsSixDeuxGain": co.gain(fonction, 0.1, 100.0),
        "ExsSixDeuxArgument": co.argument(fonction, 0.1, 100.0),
        "ExsSixDeuxNyquist": co.nyquist(fonction, 0.25, 200.0),
        "ExsSixDeuxSommet": nombre(float(pulsation), 2),
        "ExsSixDeuxArgumentAuSommet": nombre(float(argument), 2),
        "ExsSixDeuxAsymptoteDeNyquist": nombre(float(bo.asymptote_de_nyquist()), 2),
        "ExsSixDeuxVerticale": co.verticale(
            float(bo.asymptote_de_nyquist()), -8.2, 0.6
        ),
    }


def _exercice_quatre() -> dict[str, str]:
    fonction = bo.FONCTIONS["quatre"]
    pulsation, gain = bo.sommet_du_gain()
    trace = bo.trace("quatre")
    ecart = trace.gain_en_decibels(0.5) - bo.gain_reel(fonction, 0.5)
    return {
        "ExsSixQuatreAsymptote": co.asymptote(trace, 0.001, 100.0),
        "ExsSixQuatreGain": co.gain(fonction, 0.001, 100.0),
        "ExsSixQuatreSommet": nombre(pulsation, 3),
        "ExsSixQuatreGainAuSommet": nombre(gain, 2),
        "ExsSixQuatreEcart": nombre(ecart, 2),
    }


def macros_de_bode() -> dict[str, str]:
    """Les macros des exercices 1, 2 et 4."""
    return {**_associations(), **_exercice_deux(), **_exercice_quatre()}
