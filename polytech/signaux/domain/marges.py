"""Les marges de stabilite d'une boucle, lues sur sa boucle ouverte.

Planches « Boucle fermee et stabilite » du cours de signaux et systemes.
Le point `(-1, 0)` du plan de Nyquist, `(-180 degres, 0 dB)` de celui de
Black, est critique : le franchir rend la boucle instable. Deux
distances le gardent.

* La marge de phase est l'exces sur -180 degres de l'argument de la
  boucle ouverte, a la pulsation ou son module vaut un. Elle mesure le
  delai supplementaire que la boucle supporte.
* La marge de gain est l'inverse du module a la pulsation ou l'argument
  vaut -180 degres. Elle mesure le gain supplementaire admissible.

Positive pour la premiere, superieure a un pour la seconde : la boucle
est stable. Les deux se lisent ici sur la reponse frequentielle, ce qui
vaut aussi avec un delai, que nul polynome ne represente.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from polytech.signaux.domain.bode import classe_du_systeme
from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

PULSATIONS = np.logspace(-4, 4, 160_001)
"""Huit decades, vingt mille points par decade."""


@dataclass(frozen=True, slots=True)
class Marges:
    """La marge de phase en degres, la marge de gain, et leurs pulsations."""

    de_phase: float
    de_gain: float
    pulsation_de_phase: float
    pulsation_de_gain: float

    def de_gain_en_decibels(self) -> float:
        """`20 log MG`."""
        return 20 * math.log10(self.de_gain)


def _reponse(
    ouverte: sp.Expr,
) -> Callable[[NDArray[np.float64]], NDArray[np.complex128]]:
    fonction = sp.lambdify(P, ouverte, "numpy")
    return lambda pulsations: np.asarray(fonction(1j * pulsations), dtype=complex)


def arguments_deroules(
    ouverte: sp.Expr, valeurs: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """L'argument deroule, cale au depart sur `-pi/2 alpha`, en radians."""
    sans_delai = ouverte.replace(lambda e: isinstance(e, sp.exp), lambda _: sp.S.One)
    depart = -math.pi / 2 * classe_du_systeme(sans_delai)
    phases = np.unwrap(np.angle(valeurs))
    tours = round((depart - phases[0]) / (2 * math.pi))
    return np.asarray(phases + 2 * math.pi * tours, dtype=float)


def _premier_passage(ecarts: NDArray[np.float64]) -> int | None:
    """Le rang du premier passage de positif a negatif, s'il y en a un."""
    passages = np.nonzero((ecarts[:-1] >= 0) & (ecarts[1:] < 0))[0]
    return int(passages[0]) if len(passages) else None


def marges_de_stabilite(ouverte: sp.Expr) -> Marges:
    """Les deux marges de la boucle dont voici la boucle ouverte."""
    reponse = _reponse(ouverte)
    valeurs = reponse(PULSATIONS)
    phases, journal = arguments_deroules(ouverte, valeurs), np.log(PULSATIONS)
    rang = _premier_passage(np.abs(valeurs) - 1)
    if rang is None:
        msg = "le module ne vaut jamais un : la marge de phase n'est pas definie"
        raise ValueError(msg)
    bornes = (PULSATIONS[rang], PULSATIONS[rang + 1])
    de_phase = brentq(lambda w: abs(reponse(np.array([w]))[0]) - 1, *bornes)
    argument = float(np.interp(math.log(de_phase), journal, phases))
    rang = _premier_passage(phases + math.pi)
    if rang is None:
        return Marges(math.degrees(argument) + 180, math.inf, de_phase, math.inf)
    de_gain = math.exp(
        brentq(
            lambda x: float(np.interp(x, journal, phases)) + math.pi,
            *journal[rang : rang + 2],
        )
    )
    module = abs(reponse(np.array([de_gain]))[0])
    return Marges(math.degrees(argument) + 180, 1 / module, de_phase, de_gain)


def passe_a_droite_du_point_critique(ouverte: sp.Expr) -> bool:
    """Le critere simplifie dans le plan de Black : a droite de (-180 degres, 0 dB)."""
    try:
        return marges_de_stabilite(ouverte).de_phase > 0
    except ValueError:
        return bool(np.max(np.abs(_reponse(ouverte)(PULSATIONS))) < 1)
