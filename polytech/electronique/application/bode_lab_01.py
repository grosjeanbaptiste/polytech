"""Laboratoire 1 d'electronique, section 2.5 : deux traces de Bode, une lecture.

Question 7 : deux fonctions de transfert a tracer en asymptotes.
Question 8 : un diagramme de Bode donne, trois entrees sinusoidales --
la sortie a la meme pulsation, l'amplitude multipliee par `|H|` et la
phase augmentee de `arg H`.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.domain.bode_asymptotique import phase_asymptotique
from polytech.electronique.domain.phaseurs import phase_de_la_sortie

_P: Final = sp.Symbol("p")
PREMIERE: Final = (_P + 10**3) / (_P + 10**6)
SECONDE: Final = 10 * (_P + 10**3) ** 2 / (_P * (_P + 10**6))
_CASSURES: Final = {PREMIERE: ((1e3,), (1e6,)), SECONDE: ((1e3, 1e3), (1e6,))}
"""Les zeros puis les poles hors de l'origine, en rad/s."""

LECTURES: Final = {1e2: (40.0, -90.0), 1e3: (0.0, -170.0), 1e4: (-40.0, -160.0)}
"""Le gain, en dB, et la phase, en degres, lus sur la figure de l'enonce.

Les phases et deux des gains sont ceux du corrige officiel. Le gain a
`10^3 rad/s` est relu : l'officiel y lit -10 dB.
"""
ENTREES: Final = (
    (1.0, 1e2, 0.0),
    (5.0, 1e4, 0.0),
    (2.0, 1e3, math.degrees(math.pi / 6)),
)
"""Amplitude en volts, pulsation en rad/s, phase en degres (`pi / 6 = 30`)."""


def gain_exact(transfert: sp.Expr, pulsation: float) -> float:
    """`20 log |H(j w)|`."""
    valeur = complex(transfert.subs(_P, sp.I * sp.nsimplify(pulsation)))
    return 20 * math.log10(abs(valeur))


def phase_exacte(transfert: sp.Expr, pulsation: float) -> float:
    """`arg H(j w)`, en degres."""
    valeur = complex(transfert.subs(_P, sp.I * sp.nsimplify(pulsation)))
    return math.degrees(math.atan2(valeur.imag, valeur.real))


def asymptote_de_la_premiere() -> list[tuple[float, float]]:
    """Les deux cassures et le niveau, en dB, que l'asymptote y atteint."""
    bas = gain_exact(PREMIERE, 1e-3)
    return [(1e3, round(bas, 6)), (1e6, round(bas + 20 * 3, 6))]


def asymptote_de_la_seconde() -> list[tuple[float, float]]:
    """Le point de depart en `w = 1`, puis les deux cassures.

    Sous `10^3`, `H ~ 10 / p` : 20 dB en `w = 1` et -20 dB par decade.
    Le zero double redresse la pente a +20, le pole la ramene a zero.
    """
    depart = 20 * math.log10(10.0)
    creux = depart - 20 * 3
    return [(1.0, depart), (1e3, creux), (1e6, creux + 20 * 3)]


def phases_en_marches(transfert: sp.Expr) -> tuple[float, float, float]:
    """La phase asymptotique sous `10^3`, entre les cassures, au-dela de `10^6`."""
    zeros, poles_ = _CASSURES[transfert]
    origine = -90.0 if transfert is SECONDE else 0.0
    basse, mediane, haute = (
        origine + phase_asymptotique(w, zeros=zeros, poles=poles_)
        for w in (1.0, 1e4, 1e8)
    )
    return basse, mediane, haute


def amplitude_de_sortie(amplitude: float, gain_en_db: float) -> float:
    """L'amplitude d'entree multipliee par `|H|`, lu en decibels."""
    return amplitude * 10 ** (gain_en_db / 20)


def sorties() -> list[tuple[float, float]]:
    """Question 8 : l'amplitude, en volts, et la phase, en degres, de chaque sortie."""
    resultats = []
    for amplitude, pulsation, phase in ENTREES:
        gain, dephasage = LECTURES[pulsation]
        resultats.append(
            (amplitude_de_sortie(amplitude, gain), phase_de_la_sortie(phase, dephasage))
        )
    return resultats


def sortie_de_l_officiel() -> tuple[float, float]:
    """La troisieme sortie telle que l'officiel l'obtient : -10 dB et 60 degres."""
    soixante = math.degrees(math.pi / 3)  # pi/3 : ce que l'officiel lit pour pi/6
    return amplitude_de_sortie(2.0, -10.0), phase_de_la_sortie(soixante, -170.0)


def pente_entre_les_lectures() -> float:
    """La pente, en dB par decade, entre les lectures a `10^2` et a `10^4`."""
    return (LECTURES[1e4][0] - LECTURES[1e2][0]) / math.log10(1e4 / 1e2)


def gain_interpole(pulsation: float) -> float:
    """Le gain que cette droite donne a une pulsation intermediaire."""
    decades = math.log10(pulsation / 1e2)
    return LECTURES[1e2][0] + pente_entre_les_lectures() * decades
