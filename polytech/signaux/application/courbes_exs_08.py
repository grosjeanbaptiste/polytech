"""Feuille « Performances » : les courbes que le corrige trace.

Le module de `M` en fonction de la pulsation, la courbe de Black de
l'exercice 5, et les reponses indicielles simulees, retard compris.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from polytech.signaux.application import black_exs_08 as bk
from polytech.signaux.application.formatage import coordonnees
from polytech.signaux.domain.simulation_a_delai import indicielle_en_boucle_fermee
from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    from numpy.typing import NDArray

PULSATIONS_DU_TABLEAU = (0.1, 0.25, 0.35, 0.5, 0.6, 0.7)
"""Celles que l'enonce de l'exercice 5 impose."""


def module_de_m(nom: str, debut: float, fin: float) -> str:
    """`20 log |M(j w)|` pour la boucle de cet exercice, a son gain."""
    pulsations = np.logspace(np.log10(debut), np.log10(fin), 161)
    ouverte = bk.GAINS[nom] * bk.RATIONNELLES[nom] * sp.exp(-bk.RETARDS[nom] * P)
    boucle = sp.lambdify(P, ouverte, "numpy")(1j * pulsations)
    return coordonnees(pulsations, 20 * np.log10(np.abs(boucle / (1 + boucle))), 4)


def black_du_cinq(gain: float) -> str:
    """La courbe de Black de l'exercice 5, translatee de `20 log K`."""
    pulsations = np.logspace(-2, 0, 81)
    points = np.array([bk.point_de_black(float(w)) for w in pulsations])
    return coordonnees(points[:, 0], points[:, 1] + 20 * np.log10(gain), 3)


def points_du_tableau() -> str:
    """Les six points que l'enonce fait placer."""
    points = np.array([bk.point_de_black(w) for w in PULSATIONS_DU_TABLEAU])
    return coordonnees(points[:, 0], points[:, 1], 3)


def indicielle(rationnelle: sp.Expr, retard: float, duree: float) -> str:
    """La reponse indicielle de la boucle fermee, un point toutes les `duree/240` s."""
    instants: NDArray[np.float64] = np.linspace(0.0, duree, 4801)
    reponse = indicielle_en_boucle_fermee(rationnelle, retard, instants)
    return coordonnees(instants[::20], reponse[::20], 4)
