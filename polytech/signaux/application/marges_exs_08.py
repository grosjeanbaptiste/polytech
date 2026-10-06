"""Feuille « Performances », exercice 3 : un gain, un premier ordre, un retard.

La boucle ouverte vaut `2 G_c exp(-p)/(1 + p)` : le retard ne change pas
le module, il retire `omega` radians a l'argument.
"""

from __future__ import annotations

from typing import Final

import sympy as sp
from scipy.optimize import brentq

from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.erreurs_de_regime import (
    erreur_de_consigne,
    erreur_de_perturbation,
    sortie_ideale,
)
from polytech.signaux.domain.marges import marges_de_stabilite
from polytech.signaux.domain.symboles import IMAGE as P

CAPTEUR: Final = sp.Integer(2)
GAIN_DE_LA_QUESTION_TROIS: Final = sp.Rational(1, 2)
CONSIGNE: Final = sp.Integer(3)
PERTURBATION: Final = sp.Integer(5)


def _ouverte(gain: float) -> sp.Expr:
    return CAPTEUR * gain * sp.exp(-P) / (1 + P)


def marge_de_phase(gain: float) -> tuple[float, float]:
    """La marge de phase, en degres, et la pulsation ou le module vaut un."""
    marges = marges_de_stabilite(_ouverte(gain))
    return marges.de_phase, marges.pulsation_de_phase


def gain_pour_la_marge(marge: float) -> float:
    """Le gain du controleur qui donne cette marge de phase."""
    return float(brentq(lambda g: marge_de_phase(g)[0] - marge, 0.6, 1.0, xtol=1e-12))


def _boucle() -> Boucle:
    """La boucle en regime : un retard a un gain statique de un."""
    return Boucle(
        correcteur=GAIN_DE_LA_QUESTION_TROIS,
        actionneur=sp.Integer(1),
        procede=1 / (1 + P),
        capteur=CAPTEUR,
    )


def decomposition() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """La sortie ideale, l'erreur de consigne, l'erreur de perturbation."""
    boucle = _boucle()
    return (
        sortie_ideale(boucle, CONSIGNE),
        erreur_de_consigne(boucle, CONSIGNE),
        erreur_de_perturbation(boucle, PERTURBATION),
    )


def sortie_finale() -> sp.Expr:
    """`y = y_i + e_C + e_P`, comme les reponses officielles l'ecrivent."""
    return sp.Add(*decomposition())
