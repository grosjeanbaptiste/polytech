"""Feuille « Performances », exercices 6 et 7 : lire une courbe, puis regler `K`.

L'exercice 6 ne donne pas son controleur : seules les lectures de
l'abaque sont disponibles. L'exercice 7 donne tout, et se calcule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp
from scipy.optimize import brentq

from polytech.signaux.application.black_exs_08 import Estimation, estimer
from polytech.signaux.domain.fonction_m import caracteristiques_de_m
from polytech.signaux.domain.marges import marges_de_stabilite
from polytech.signaux.domain.simulation_a_delai import indicielle_en_boucle_fermee
from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

LECTURES_DU_SIX: Final = (0.0, 2.3, 0.24)
"""`|M(0)|` et `|M|max` en decibels, la bande passante en rad/s : lus sur la figure."""
RATIONNELLE: Final = 2 / ((1 + 2 * P) * (1 + 5 * P) * (1 + 20 * P))
RETARD: Final = 10.0
GAIN_DES_QUESTIONS_TROIS_ET_QUATRE: Final = sp.Rational(1, 2)


def estimation_six() -> Estimation:
    """Les regles du cours appliquees aux trois lectures de l'exercice 6."""
    en_continu, maximum, bande = LECTURES_DU_SIX
    return estimer(en_continu_db=en_continu, maximum_db=maximum, bande=bande)


def _ouverte(gain: float) -> sp.Expr:
    return gain * RATIONNELLE * sp.exp(-RETARD * P)


def marge_et_gain_limite() -> tuple[float, float]:
    """La marge de gain pour `K = 1`, en decibels, et le `K` qui l'epuise."""
    marges = marges_de_stabilite(_ouverte(1.0))
    return marges.de_gain_en_decibels(), marges.de_gain


def boucle_a_la_limite() -> complex:
    """`K_max G(j w)` a la pulsation ou l'argument vaut -180 degres."""
    marges = marges_de_stabilite(_ouverte(1.0))
    valeur = _ouverte(marges.de_gain).subs(P, sp.I * marges.pulsation_de_gain)
    return complex(valeur)


def estimation_sept(gain: float) -> Estimation:
    """L'estimation de la reponse indicielle de l'exercice 7 pour ce `K`."""
    m = caracteristiques_de_m(_ouverte(gain))
    return estimer(
        en_continu_db=m.en_continu_en_decibels(),
        maximum_db=m.maximum_en_decibels(),
        bande=m.bande_passante,
    )


def gain_pour_le_depassement(depassement: float) -> float:
    """Le `K` pour lequel la regle `9 + 6 ecart` annonce ce depassement."""

    def ecart(gain: float) -> float:
        return estimation_sept(gain).depassement - depassement

    return float(brentq(ecart, 0.35, 1.0, xtol=1e-4))


def sortie_finale(*, consigne: object, perturbation: object) -> sp.Expr:
    """`y = (K G(0) r - p)/(1 + K G(0))` : la perturbation entre en sortie."""
    statique = GAIN_DES_QUESTIONS_TROIS_ET_QUATRE * RATIONNELLE.subs(P, 0)
    return (statique * sp.sympify(consigne) - sp.sympify(perturbation)) / (1 + statique)


def amplitude_finale(gain: float, instants: NDArray[np.float64]) -> float:
    """L'excursion de la reponse indicielle sur le dernier cinquieme de l'horizon."""
    reponse = indicielle_en_boucle_fermee(gain * RATIONNELLE, RETARD, instants)
    fin = reponse[4 * len(reponse) // 5 :]
    return float(fin.max() - fin.min())
