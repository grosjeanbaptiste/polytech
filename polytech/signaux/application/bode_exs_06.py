"""Feuille « Analyse harmonique », exercices 1, 2 et 4 : les courbes de Bode.

Le trace asymptotique se lit sur la forme de Bode : un gain, une classe,
des cassures. Ce module met chaque transmittance de la feuille sous
cette forme, et garde a cote la reponse exacte pour les comparer.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import PULSATION as W
from polytech.signaux.domain.trace_asymptotique import trace_d_une_transmittance

if TYPE_CHECKING:
    from polytech.signaux.domain.bode import TraceAsymptotique

_DEMI: Final = sp.Rational(1, 2)
FONCTIONS: Final[dict[str, sp.Expr]] = {
    "a": 10 / ((1 + P / 3) * (1 + P / 2)),
    "b": 25 * (P + 1) / (P * (P + 5)),
    "c": sp.Rational(2, 5)
    * P
    * (P + 2)
    / ((P + sp.Rational(1, 5)) * (P + 1) * (1 + P / 10)),
    "d": 2 * (1 + P / 10) / (P * (1 + P / 100) * (1 + P / 1000) * (1 + P / 2000)),
    "e": 16 * P / ((P + 1) * (P + 2)),
    "deux": 10 * (P + 1) / (P * (P + 5)),
    "quatre": 10 * P / ((20 * P + 1) * (P + _DEMI)),
}
"""Les cinq transmittances de l'exercice 1, puis celles des exercices 2 et 4."""

COURBES: Final[dict[int, tuple[int, ...]]] = {
    1: (0, -20, -40),
    2: (20, 0, -20),
    3: (-20, 0, -20),
    4: (-20, 0, -20, -40, -60),
    5: (20, 0, -20, 0, -20),
}
"""Les pentes successives des cinq courbes de la feuille, lues sur ses figures."""


def trace(nom: str) -> TraceAsymptotique:
    """Le trace asymptotique de cette transmittance de la feuille."""
    return trace_d_une_transmittance(FONCTIONS[nom])


def association() -> dict[str, int]:
    """Pour chaque transmittance de l'exercice 1, la courbe qui a ses pentes."""
    par_pentes = {pentes: numero for numero, pentes in COURBES.items()}
    return {nom: par_pentes[trace(nom).pentes()] for nom in "abcde"}


def reponse(transmittance: sp.Expr, pulsation: float) -> complex:
    """`H(j w)`, en nombre complexe."""
    return complex(transmittance.subs(P, sp.I * pulsation))


def gain_reel(transmittance: sp.Expr, pulsation: float) -> float:
    """`20 log |H(j w)|`, en decibels : la courbe reelle."""
    return 20 * math.log10(abs(reponse(transmittance, pulsation)))


def argument_reel(transmittance: sp.Expr, pulsation: float) -> float:
    """`arg H(j w)`, en degres, entre -180 et 180."""
    return math.degrees(cmath.phase(reponse(transmittance, pulsation)))


def point_de_nyquist(pulsation: float) -> tuple[float, float]:
    """Les parties reelle et imaginaire de la transmittance de l'exercice 2."""
    valeur = reponse(FONCTIONS["deux"], pulsation)
    return valeur.real, valeur.imag


def asymptote_de_nyquist() -> sp.Expr:
    """La limite de la partie reelle quand la pulsation tend vers zero."""
    frequentielle = FONCTIONS["deux"].subs(P, sp.I * W)
    return sp.limit(sp.re(sp.simplify(frequentielle)), W, 0)


def sommet_de_l_argument() -> tuple[sp.Expr, sp.Expr]:
    """La pulsation ou l'argument de l'exercice 2 culmine, et sa valeur en degres."""
    argument = -sp.pi / 2 + sp.atan(W) - sp.atan(W / 5)
    (pulsation,) = sp.solve(sp.diff(argument, W), W)
    return pulsation, argument.subs(W, pulsation) * 180 / sp.pi


def sommet_du_gain() -> tuple[float, float]:
    """La pulsation ou le module de l'exercice 4 culmine, et son gain en decibels."""
    frequentielle = FONCTIONS["quatre"].subs(P, sp.I * W)
    carre = sp.simplify(frequentielle * sp.conjugate(frequentielle))
    (pulsation,) = sp.solve(sp.numer(sp.together(sp.diff(carre, W))), W)
    return float(pulsation), gain_reel(FONCTIONS["quatre"], float(pulsation))
