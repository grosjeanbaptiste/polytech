"""Feuille « Performances », exercices 4 et 5 : de la courbe de Black a la reponse.

La fonction `M = L/(1 + L)` se lit sur l'abaque ; quatre nombres la
resument, et trois regles en tirent l'allure de la reponse indicielle.
Ici les memes nombres sont calcules, puis la reponse est simulee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from polytech.signaux.domain.fonction_m import (
    CaracteristiquesDeM,
    caracteristiques_de_m,
    depassement_estime,
    temps_de_montee_estime,
)
from polytech.signaux.domain.marges import marges_de_stabilite
from polytech.signaux.domain.simulation_a_delai import indicielle_en_boucle_fermee
from polytech.signaux.domain.symboles import IMAGE as P

RATIONNELLES: Final[dict[str, sp.Expr]] = {
    "quatre": sp.Rational(48, 100) / ((P + 2) * (P + 1) * (P + sp.Rational(1, 10))),
    "cinq": 1 / ((1 + 5 * P) * (1 + P / 2)),
}
"""Les boucles ouvertes sans retard ni gain reglable : exercices 4 et 5."""
RETARDS: Final = {"quatre": 5.0, "cinq": 3.0}
GAINS: Final = {"quatre": 1.0, "cinq": 2.0}
"""Le gain auquel chaque exercice demande la reponse indicielle."""
SEUIL_DE_DEPART: Final = 1e-9
"""Au-dessus : la reponse a quitte zero."""


@dataclass(frozen=True, slots=True)
class Estimation:
    """Ce que les regles du cours tirent de `M` pour un echelon unite."""

    finale: float
    depassement: float
    maximum: float
    montee: tuple[float, float]


def _ouverte(nom: str, gain: float | None = None) -> sp.Expr:
    gain = GAINS[nom] if gain is None else gain
    return gain * RATIONNELLES[nom] * sp.exp(-RETARDS[nom] * P)


def caracteristiques(nom: str) -> CaracteristiquesDeM:
    """`|M(0)|`, `|M|max` et la bande passante, calcules et non lus."""
    return caracteristiques_de_m(_ouverte(nom))


def estimer(*, en_continu_db: float, maximum_db: float, bande: float) -> Estimation:
    """Les trois regles : valeur finale, `9 + 6 ecart`, `0,3 < t_m BP < 0,45`."""
    finale = 10 ** (en_continu_db / 20)
    depassement = depassement_estime(maximum_db - en_continu_db)
    montee = temps_de_montee_estime(bande / (2 * math.pi))
    return Estimation(finale, depassement, finale * (1 + depassement / 100), montee)


def estimation(nom: str) -> Estimation:
    """L'estimation, a partir des caracteristiques calculees."""
    m = caracteristiques(nom)
    return estimer(
        en_continu_db=m.en_continu_en_decibels(),
        maximum_db=m.maximum_en_decibels(),
        bande=m.bande_passante,
    )


def simulation(nom: str) -> tuple[float, float, float]:
    """La valeur finale, le maximum et l'instant de demarrage de la reponse simulee."""
    instants = np.linspace(0.0, 300.0, 6001)
    rationnelle = GAINS[nom] * RATIONNELLES[nom]
    reponse = indicielle_en_boucle_fermee(rationnelle, RETARDS[nom], instants)
    depart = float(instants[np.nonzero(reponse > SEUIL_DE_DEPART)[0][0] - 1])
    return float(reponse[-1]), float(reponse.max()), depart


def point_de_black(pulsation: float) -> tuple[float, float]:
    """Exercice 5, `K = 1` : l'argument deroule, en degres, et le gain en decibels."""
    argument = -3 * pulsation - math.atan(5 * pulsation) - math.atan(pulsation / 2)
    module = 1 / math.sqrt((1 + 25 * pulsation**2) * (1 + pulsation**2 / 4))
    return math.degrees(argument), 20 * math.log10(module)


def marge_de_gain_en_decibels(gain: float) -> float:
    """Exercice 5 : la marge de gain pour ce `K`.

    Elle se mesure sur la boucle de gain 2, dont le module croise un, puis
    se ramene au gain demande : un gain ne fait que translater la courbe.
    """
    reference = GAINS["cinq"]
    limite = marges_de_stabilite(_ouverte("cinq", reference)).de_gain * reference
    return 20 * math.log10(limite / gain)


def module_maximal_de_la_boucle(gain: float) -> float:
    """Exercice 5 : le plus grand `|K G(j w)|`, atteint en `w = 0`."""
    pulsations = np.logspace(-6, 2, 801)
    return float(
        max(10 ** (point_de_black(float(w))[1] / 20) for w in pulsations) * gain
    )


def gain_pour_la_marge(marge: float) -> float:
    """Exercice 5 : le `K` qui donne cette marge de phase."""

    def ecart(gain: float) -> float:
        return marges_de_stabilite(_ouverte("cinq", gain)).de_phase - marge

    return float(brentq(ecart, 1.05, 4.0, xtol=1e-9))
