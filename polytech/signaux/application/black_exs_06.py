"""Feuille « Analyse harmonique », exercices 3 et 5 : le plan de Black-Nichols.

Black porte l'argument en abscisse et le gain en ordonnee : une courbe
s'y reconnait a l'argument d'ou elle part, et au cote vers lequel elle
s'ecarte avant d'y revenir.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from scipy.optimize import brentq, minimize_scalar

from polytech.signaux.application.bode_exs_06 import argument_reel, gain_reel
from polytech.signaux.domain.marges import Marges, marges_de_stabilite
from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    import sympy as sp

TROIS: Final = 5 / (P * (1 + P / 2))
"""La transmittance de l'exercice 3."""
FONCTIONS: Final[dict[str, sp.Expr]] = {
    "T1": (1 + 2 * P) / (P * (1 + 5 * P)),
    "T2": (1 + 10 * P) / (P**2 * (1 + P)),
    "T3": (1 + 20 * P) / (P * (1 + 5 * P)),
}
COURBES: Final[dict[str, tuple[int, str]]] = {
    "A": (-90, "droite"),
    "B": (-180, "droite"),
    "C": (-90, "gauche"),
}
"""Pour chaque courbe de la feuille : l'argument de ses deux bouts, et le cote
vers lequel elle s'en ecarte."""
_CLASSES: Final = {"T1": 1, "T2": 2, "T3": 1}
"""Le nombre d'integrateurs : l'argument de depart vaut `-90` degres par classe."""


def point_de_black(transmittance: sp.Expr, pulsation: float) -> tuple[float, float]:
    """`(argument en degres, gain en decibels)` a cette pulsation."""
    return argument_reel(transmittance, pulsation), gain_reel(transmittance, pulsation)


def passage_a_zero_decibel() -> tuple[float, float]:
    """La pulsation ou la courbe de l'exercice 3 coupe 0 dB, et son argument."""
    pulsation = brentq(lambda w: gain_reel(TROIS, w), 0.1, 100.0, xtol=1e-12)
    return pulsation, point_de_black(TROIS, pulsation)[0]


def marges() -> Marges:
    """Les marges de stabilite de la boucle fermee sur l'exercice 3."""
    return marges_de_stabilite(TROIS)


def extreme_de_l_argument(nom: str) -> float:
    """L'argument, en degres, la ou la courbe s'ecarte le plus de son depart."""
    depart = -90 * _CLASSES[nom]

    def ecart(exposant: float) -> float:
        return -abs(point_de_black(FONCTIONS[nom], 10**exposant)[0] - depart)

    optimum = minimize_scalar(ecart, bounds=(-3, 3), method="bounded")
    return point_de_black(FONCTIONS[nom], 10**optimum.x)[0]


def association() -> dict[str, str]:
    """Pour chaque transmittance de l'exercice 5, la courbe de la feuille."""
    par_allure = {allure: lettre for lettre, allure in COURBES.items()}
    resultat = {}
    for nom in FONCTIONS:
        depart = -90 * _CLASSES[nom]
        cote = "droite" if extreme_de_l_argument(nom) > depart else "gauche"
        resultat[nom] = par_allure[(depart, cote)]
    return resultat
