"""Seance « Chap. 4-5 », exercice 4 : transmittances, reponses et diagrammes.

Une reponse indicielle se reconnait a sa valeur de depart, a sa valeur
finale et a ses oscillations ; un diagramme, a ses poles et a son zero.
Les deux tableaux ci-dessous sont lus sur les figures de la feuille.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P

FONCTIONS: Final[dict[str, sp.Expr]] = {
    "A": 1 / (P + 1),
    "B": P / (P + 1),
    "C": 4 / (P**2 + P + 4),
    "D": 16 / (P**2 + 2 * P + 16),
    "E": P / (P**2 + P + 4),
}
REPONSES: Final[dict[str, tuple[int, int, str]]] = {
    "a": (0, 1, "lente"),
    "b": (1, 0, "aucune"),
    "c": (0, 0, "lente"),
    "d": (0, 1, "rapide"),
    "e": (0, 1, "aucune"),
}
"""Pour chaque reponse de la feuille : depart, arrivee, oscillation."""
DIAGRAMMES: Final[dict[int, tuple[int, bool, str]]] = {
    1: (2, True, "lente"),
    2: (2, False, "lente"),
    3: (1, False, "aucune"),
    4: (2, False, "rapide"),
    5: (1, True, "aucune"),
}
"""Pour chaque diagramme : nombre de poles, zero a l'origine, oscillation."""
SEUIL_DE_RAPIDITE: Final = 3
"""En rad/s : au-dela, la feuille montre des oscillations serrees."""


def _oscillation(transmittance: sp.Expr) -> str:
    poles = sp.roots(sp.fraction(transmittance)[1], P)
    pulsation = max(abs(sp.im(pole)) for pole in poles)
    if pulsation == 0:
        return "aucune"
    return "rapide" if pulsation > SEUIL_DE_RAPIDITE else "lente"


def _allure(transmittance: sp.Expr) -> tuple[int, int, str]:
    """Depart et arrivee de la reponse indicielle, par les deux theoremes."""
    depart = sp.limit(transmittance, P, sp.oo)
    return int(depart), int(transmittance.subs(P, 0)), _oscillation(transmittance)


def _poles_et_zero(transmittance: sp.Expr) -> tuple[int, bool, str]:
    numerateur, denominateur = sp.fraction(transmittance)
    poles = sum(sp.roots(denominateur, P).values())
    return poles, numerateur.subs(P, 0) == 0, _oscillation(transmittance)


def association() -> dict[str, tuple[str, int]]:
    """Pour chaque transmittance : sa reponse indicielle et son diagramme."""
    par_allure = {allure: lettre for lettre, allure in REPONSES.items()}
    par_poles = {poles: numero for numero, poles in DIAGRAMMES.items()}
    return {
        nom: (par_allure[_allure(h)], par_poles[_poles_et_zero(h)])
        for nom, h in FONCTIONS.items()
    }


def reponses_et_diagrammes() -> dict[str, int]:
    """La forme de la feuille : a chaque reponse indicielle, son diagramme."""
    return dict(sorted(association().values()))


def _amortissement_et_pulsation(nom: str) -> tuple[float, float]:
    _, un, zero = (
        float(c) for c in sp.Poly(sp.fraction(FONCTIONS[nom])[1], P).all_coeffs()
    )
    propre = math.sqrt(zero)
    return un / (2 * propre), propre


def depassement(nom: str) -> float:
    """Le depassement relatif de la reponse indicielle d'un second ordre."""
    amortissement, _ = _amortissement_et_pulsation(nom)
    return math.exp(-math.pi * amortissement / math.sqrt(1 - amortissement**2))


def periode(nom: str) -> float:
    """La periode des oscillations amorties, en secondes."""
    amortissement, propre = _amortissement_et_pulsation(nom)
    return 2 * math.pi / (propre * math.sqrt(1 - amortissement**2))
