"""Recueil d'exercices, planche 1 : apparier entrees et reponses impulsionnelles.

Deux entrees -- un echelon `x1`, une sinusoide `x2` -- et deux reponses
impulsionnelles -- une exponentielle breve `h1`, une oscillation amortie
`h2`, accordee sur `x2`. La planche n'a pas d'echelle : les valeurs
ci-dessous en reproduisent les allures, et rien de plus.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]

PAS: Final = 2e-3
DUREE: Final = 40.0
PERIODE: Final = 1.0
"""Celle de `x2` et des oscillations de `h2`, en secondes."""
TAUX_DE_H1: Final = 2.0
"""`h1 = a e^(-a t)` : eteinte en deux periodes, d'aire un."""
AMORTISSEMENT_DE_H2: Final = 0.2
"""`h2 = e^(-s t) sin(w t)` : une vingtaine de periodes pour s'eteindre."""
SEUIL: Final = 0.1
"""La part de l'amplitude maximale en deca de laquelle on dit « rien »."""
PERIODE_TEMOIN: Final = 3
"""Le rang de la periode ou l'on regarde si le transitoire dure encore."""


def _signaux() -> dict[str, Signal]:
    instants = np.arange(0.0, DUREE, PAS)
    pulsation = 2 * np.pi / PERIODE
    return {
        "x1": np.ones_like(instants),
        "x2": np.sin(pulsation * instants),
        "h1": TAUX_DE_H1 * np.exp(-TAUX_DE_H1 * instants),
        "h2": np.exp(-AMORTISSEMENT_DE_H2 * instants) * np.sin(pulsation * instants),
    }


def sortie(entree: str, reponse: str) -> Signal:
    """`y = x * h`, l'integrale de convolution echantillonnee."""
    signaux = _signaux()
    complet = np.convolve(signaux[entree], signaux[reponse]) * PAS
    return complet[: signaux[entree].size]


def _excursion(signal: Signal, rang: int) -> float:
    """L'ecart crete a crete sur la periode de ce rang."""
    par_periode = round(PERIODE / PAS)
    return float(np.ptp(signal[rang * par_periode : (rang + 1) * par_periode]))


def traits(signal: Signal) -> tuple[bool, bool]:
    """`(oscille en regime, le transitoire dure)`.

    Le regime se lit sur la derniere periode ; le transitoire dure si, a
    la periode temoin, l'excursion n'est pas encore celle du regime.
    """
    echelle = float(np.max(np.abs(signal)))
    finale = _excursion(signal, round(DUREE / PERIODE) - 1)
    temoin = _excursion(signal, PERIODE_TEMOIN)
    return finale > SEUIL * echelle, abs(temoin - finale) > SEUIL * echelle


def appariement() -> dict[str, tuple[str, str]]:
    """Pour chaque sortie de la planche, l'entree et la reponse qui la donnent."""
    par_traits = {(False, False): "y1", (True, False): "y2"}
    par_traits |= {(False, True): "y3", (True, True): "y4"}
    return {
        par_traits[traits(sortie(entree, reponse))]: (entree, reponse)
        for entree in ("x1", "x2")
        for reponse in ("h1", "h2")
    }
