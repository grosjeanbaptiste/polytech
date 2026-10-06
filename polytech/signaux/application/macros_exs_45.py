"""Macros de la seance « Chap. 4-5 ». Prefixe `ExsQuatreCinq`."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy import signal

from polytech.signaux.application import amortie_exs_45 as am
from polytech.signaux.application import association_exs_45 as asso
from polytech.signaux.application import circuits_exs_45 as ci
from polytech.signaux.application import systemes_exs_45 as sy
from polytech.signaux.application.formatage import coordonnees, nombre
from polytech.signaux.domain.modeles import coefficients
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transmittance import Transmittance


def _trace(
    expression: sp.Expr,
    fin: float,
    points: int = 201,
    *,
    en_millisecondes: bool = False,
) -> str:
    """Les points d'une courbe ; l'abscisse en millisecondes si on le demande."""
    instants = np.linspace(0.0, fin, points)
    valeurs = sp.lambdify(T, expression, "numpy")(instants)
    abscisses = instants * 1e3 if en_millisecondes else instants
    return coordonnees(abscisses, np.broadcast_to(valeurs, instants.shape), 4)


def _amorties() -> dict[str, str]:
    sinusoide = sp.exp(-am.SIGMA * T) * sp.cos(am.PULSATION * T + am.PHASE)
    enveloppe = sp.exp(-am.SIGMA * T)
    return {
        "ExsQuatreCinqUnSignal": _trace(sinusoide, 0.2, 801, en_millisecondes=True),
        "ExsQuatreCinqUnEnveloppe": _trace(enveloppe, 0.2, 41, en_millisecondes=True),
        "ExsQuatreCinqUnDepart": nombre(am.reperes_de_la_sinusoide().depart, 3),
        "ExsQuatreCinqDeuxSignal": _trace(am.original(), 7.0),
        "ExsQuatreCinqDeuxPeriode": nombre(float(am.duree_et_periode()[1]), 2),
    }


def _systemes() -> dict[str, str]:
    h = sy.impulsionnelle_du_graphe()
    return {
        "ExsQuatreCinqTroisImpulsionnelle": _trace(h, 0.6, en_millisecondes=True),
        "ExsQuatreCinqTroisZero": nombre(float(sy.premier_zero_du_graphe()) * 1e3, 1),
        "ExsQuatreCinqSeptLimite": nombre(float(sy.alpha_limite()), 2),
    }


def _associations() -> dict[str, str]:
    macros = {}
    instants = np.linspace(0.0, 10.0, 201)
    for nom, fonction in asso.FONCTIONS.items():
        numerateur, denominateur = coefficients(Transmittance(expression=fonction))
        _, reponse = signal.step((numerateur, denominateur), T=instants)
        macros[f"ExsQuatreCinqQuatre{nom}"] = coordonnees(instants, reponse, 4)
    macros["ExsQuatreCinqQuatreDepassement"] = nombre(100 * asso.depassement("C"), 1)
    for nom, suffixe in (("C", "Lente"), ("D", "Rapide")):
        pulsation = 2 * math.pi / asso.periode(nom)
        macros[f"ExsQuatreCinqQuatrePulsation{suffixe}"] = nombre(pulsation, 2)
    return macros


def _circuits() -> dict[str, str]:
    reperes = ci.reperes_du_huit()
    return {
        "ExsQuatreCinqHuitIndicielle": _trace(
            ci.indicielle_du_huit(), 0.008, en_millisecondes=True
        ),
        "ExsQuatreCinqHuitDepassement": nombre(100 * float(reperes.depassement), 1),
        "ExsQuatreCinqHuitMaximum": nombre(
            float(reperes.finale) * (1 + math.exp(-math.pi)), 3
        ),
    }


def macros_de_la_seance() -> dict[str, str]:
    """Les macros des huit exercices."""
    return {**_amorties(), **_systemes(), **_associations(), **_circuits()}
