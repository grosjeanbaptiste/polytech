"""Macros du recueil d'exercices, planches sur les systemes. Prefixe `Recueil`."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy import signal

from polytech.signaux.application import recueil_circuits as ci
from polytech.signaux.application import recueil_graphes as gr
from polytech.signaux.application import recueil_poles as po
from polytech.signaux.application import recueil_trottinette as tr
from polytech.signaux.application import systemes_exs_45 as sy
from polytech.signaux.application.formatage import coordonnees, nombre
from polytech.signaux.domain.symboles import TEMPS as T


def _amorties() -> dict[str, str]:
    instants = np.linspace(0.0, 0.05, 501)
    enveloppe = np.exp(-po.SIGMA * instants)
    sinusoide = enveloppe * np.cos(po.PULSATION * instants + po.PHASE)
    secondes = np.linspace(0.0, 7.0, 141)
    tension = np.exp(-secondes) * (np.cos(secondes) - np.sin(secondes))
    return {
        "RecueilQuatreSignal": coordonnees(instants * 1e3, sinusoide, 4),
        "RecueilQuatreEnveloppe": coordonnees(instants[::10] * 1e3, enveloppe[::10], 4),
        "RecueilQuatreDepart": nombre(po.reperes_de_la_sinusoide()[2], 3),
        "RecueilQuatreReste": nombre(100 * po.enveloppe_apres(periodes=1), 0),
        "RecueilCinqSignal": coordonnees(secondes, tension, 4),
        "RecueilCinqZero": nombre(math.pi / 4, 3),
        "RecueilCinqPeriode": nombre(2 * math.pi, 2),
    }


def _poles() -> dict[str, str]:
    millisecondes = np.linspace(0.0, 600.0, 201)
    h = sp.lambdify(T, sy.impulsionnelle_du_graphe(), "numpy")(millisecondes / 1e3)
    sigma, omega = po.poles_lus_sur_la_reponse()
    lent, rapide = po.poles_des_deux_cellules()
    constante, extinction = po.extinction()
    return {
        "RecueilNeufLimite": nombre(float(gr.alpha_limite()), 2),
        "RecueilDixImpulsionnelle": coordonnees(millisecondes, h, 4),
        "RecueilDixZero": nombre(1e3 * math.pi / 40, 1),
        "RecueilOnzeSigma": nombre(sigma, 2),
        "RecueilOnzeOmega": nombre(omega, 2),
        "RecueilDouzeLent": nombre(lent, 1),
        "RecueilDouzeRapide": nombre(rapide, 1),
        "RecueilDouzeConstante": nombre(constante * 1e3, 1),
        "RecueilDouzeExtinction": nombre(extinction * 1e3, 0),
        "RecueilQuatorzeK": nombre(float(gr.reglages()[gr.K]), 1),
        "RecueilQuatorzeM": nombre(float(gr.reglages()[gr.M]), 1),
    }


def _reglages() -> dict[str, str]:
    reduits = np.linspace(0.0, 5.0, 101)
    exact, approche = float(tr.gamma_exact()), float(tr.gamma_approche())
    instants, reponse = signal.step(
        ([1.0], [1.0, exact, 1.0]), T=np.linspace(0, 16, 321)
    )
    return {
        "RecueilQuinzeIndicielle": coordonnees(reduits, 1 - np.exp(-reduits) / 2, 4),
        "RecueilQuinzeEnTau": nombre(1 - math.exp(-1) / 2, 3),
        "RecueilSeizeExact": nombre(exact, 3),
        "RecueilSeizeApproche": nombre(approche, 3),
        "RecueilSeizeEcart": nombre(100 * (1 - approche / exact), 1),
        "RecueilSeizeOscillations": nombre(
            tr.oscillations_avant_le_regime(tr.gamma_approche()), 2
        ),
        "RecueilSeizeDuree": nombre(10 / exact, 1),
        "RecueilSeizeIndicielle": coordonnees(instants, reponse, 4),
        "RecueilDixNeufLimite": nombre(float(ci.resistance_limite_19()), 1),
    }


def macros_du_recueil_systemes() -> dict[str, str]:
    """Les macros des planches 4, 5, 9 a 12, 14 a 16 et 19."""
    return {**_amorties(), **_poles(), **_reglages()}
