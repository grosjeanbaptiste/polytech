"""Feuille « Analyse harmonique » : les points des courbes que le corrige trace.

Une asymptote se trace par ses sommets ; une courbe reelle, par un
balayage logarithmique des pulsations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from polytech.signaux.application.bode_exs_06 import (
    argument_reel,
    gain_reel,
    reponse,
)
from polytech.signaux.application.formatage import coordonnees

if TYPE_CHECKING:
    import sympy as sp

    from polytech.signaux.domain.bode import TraceAsymptotique

POINTS = 121
"""Par courbe reelle : assez pour une courbe lisse, peu pour le fichier."""


def _balayage(debut: float, fin: float) -> np.ndarray:
    return np.logspace(np.log10(debut), np.log10(fin), POINTS)


def asymptote(trace: TraceAsymptotique, debut: float, fin: float) -> str:
    """Les sommets du trace asymptotique entre ces deux pulsations."""
    cassures = sorted({w for w, _ in trace.cassures if debut < w < fin})
    pulsations = np.array([debut, *cassures, fin])
    gains = np.array([trace.gain_en_decibels(float(w)) for w in pulsations])
    return coordonnees(pulsations, gains, 4)


def gain(transmittance: sp.Expr, debut: float, fin: float) -> str:
    """La courbe de gain reelle, en decibels."""
    pulsations = _balayage(debut, fin)
    gains = np.array([gain_reel(transmittance, float(w)) for w in pulsations])
    return coordonnees(pulsations, gains, 4)


def argument(transmittance: sp.Expr, debut: float, fin: float) -> str:
    """La courbe d'argument reelle, en degres."""
    pulsations = _balayage(debut, fin)
    arguments = np.array([argument_reel(transmittance, float(w)) for w in pulsations])
    return coordonnees(pulsations, arguments, 4)


def nyquist(transmittance: sp.Expr, debut: float, fin: float) -> str:
    """La courbe de Nyquist : partie reelle, partie imaginaire."""
    valeurs = np.array(
        [reponse(transmittance, float(w)) for w in _balayage(debut, fin)]
    )
    return coordonnees(valeurs.real, valeurs.imag, 4)


def black(transmittance: sp.Expr, debut: float, fin: float) -> str:
    """La courbe de Black : argument en degres, gain en decibels."""
    pulsations = _balayage(debut, fin)
    arguments = np.array([argument_reel(transmittance, float(w)) for w in pulsations])
    gains = np.array([gain_reel(transmittance, float(w)) for w in pulsations])
    return coordonnees(arguments, gains, 3)


def point(abscisse: float, ordonnee: float) -> str:
    """Un seul point, a porter par un marqueur."""
    return coordonnees(np.array([abscisse]), np.array([ordonnee]), 1)


def verticale(abscisse: float, bas: float, haut: float) -> str:
    """Les deux bouts d'une droite verticale."""
    return coordonnees(np.array([abscisse, abscisse]), np.array([bas, haut]), 1)
