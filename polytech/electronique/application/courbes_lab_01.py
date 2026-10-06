"""Laboratoire 1 d'electronique : les courbes que le corrige trace.

Chaque courbe est le module, en decibels, d'une fonction de transfert de
la seance, sur la plage que la manipulation balaie. Elles sont evaluees
numeriquement sur le transfert lui-meme.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import sympy as sp

from polytech.electronique.application import bode_lab_01 as bo
from polytech.electronique.application import rc_lab_01 as rc
from polytech.electronique.application import rlc_lab_01 as rl

_P: Final = sp.Symbol("p")
POINTS_PAR_DECADE: Final = 60


def _gains(transfert: sp.Expr, pulsations: np.ndarray) -> np.ndarray:
    """`20 log |H(j w)|` sur une grille de pulsations."""
    numerique = sp.lambdify(_P, transfert, "numpy")
    gains: np.ndarray = 20 * np.log10(np.abs(numerique(1j * pulsations)))
    return gains


def _grille(basse: float, haute: float) -> np.ndarray:
    decades = math.log10(haute / basse)
    return np.logspace(
        math.log10(basse), math.log10(haute), round(decades * POINTS_PAR_DECADE) + 1
    )


def passe_haut() -> tuple[np.ndarray, np.ndarray]:
    """Le RC passe-haut, de 10 Hz a 100 kHz : frequences et gains."""
    frequences = _grille(10.0, 1e5)
    transfert = rc.passe_haut(sp.Float(rc.RESISTANCE), sp.Float(rc.CAPACITE), _P)
    return frequences, _gains(transfert, 2 * math.pi * frequences)


def asymptote_du_passe_haut() -> list[tuple[float, float]]:
    """+20 dB par decade jusqu'a la coupure, puis 0 dB."""
    coupure = rc.frequence_de_coupure()
    return [(10.0, 20 * math.log10(10.0 / coupure)), (coupure, 0.0), (1e5, 0.0)]


def passe_bas(resistance: float) -> tuple[np.ndarray, np.ndarray]:
    """Le RLC passe-bas vu depuis `V_1`, de 100 Hz a 100 kHz.

    La frequence de resonance est ajoutee a la grille : le sommet d'un
    circuit peu amorti est plus etroit que le pas, et il serait ecrete.
    """
    frequences = np.sort(np.append(_grille(100.0, 1e5), rl.frequence_de_resonance()))
    composants = (sp.Float(v) for v in (resistance, rl.INDUCTANCE, rl.CAPACITE))
    r, ind, c = composants
    transfert = rl.passe_bas(r, ind, c, _P)
    return frequences, _gains(transfert, 2 * math.pi * frequences)


def bode(transfert: sp.Expr) -> tuple[np.ndarray, np.ndarray]:
    """Une fonction de la question 7, de 1 a `10^8 rad/s`."""
    pulsations = _grille(1.0, 1e8)
    return pulsations, _gains(transfert, pulsations)


def asymptotes_de_la_question_7() -> tuple[list[tuple[float, float]], ...]:
    """Les lignes brisees des deux fonctions, prolongees jusqu'aux bords."""
    (z, bas), (q, haut) = bo.asymptote_de_la_premiere()
    premiere = [(1.0, bas), (z, bas), (q, haut), (1e8, haut)]
    seconde = [
        *bo.asymptote_de_la_seconde(),
        (1e8, bo.asymptote_de_la_seconde()[-1][1]),
    ]
    return premiere, seconde
