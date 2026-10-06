"""TP « Reservoir », troisieme partie : le controleur PID de l'enonce.

`G_c = 0,2`, `T_i = T_d = 400` s, `lambda = 0,1`. L'action integrale
annule l'erreur de regime ; l'avance de phase rend de la marge.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import sympy as sp
from scipy import signal

from polytech.signaux.application import cuves_boucle as bo
from polytech.signaux.application.black_exs_08 import Estimation, estimer
from polytech.signaux.domain.fonction_m import (
    CaracteristiquesDeM,
    caracteristiques_de_m,
)
from polytech.signaux.domain.marges import Marges, marges_de_stabilite
from polytech.signaux.domain.pid_serie import PidSerie
from polytech.signaux.domain.symboles import IMAGE as P

CONTROLEUR: Final = PidSerie(
    gain=sp.Rational(1, 5),
    integrale=sp.Integer(400),
    derivee=sp.Integer(400),
    filtre=sp.Rational(1, 10),
)
OUVERTE: Final = sp.cancel(CONTROLEUR.transmittance() * bo.PROCEDE)
PERTURBATION: Final = sp.Rational(98, 100)
"""Le gain par lequel `x_B` entre dans la boucle, selon l'enonce."""
DUREE: Final = 6000.0


def regime_sous_consigne() -> tuple[sp.Expr, sp.Expr]:
    """`y` et la commande `nu`, en regime, pour un echelon unite de consigne.

    L'integrateur du controleur ne recoit rien en regime : l'ecart est
    nul, `y = 1`, et la commande est ce qu'il faut au procede pour le tenir.
    """
    sortie = sp.Integer(1)
    return sortie, sortie / bo.PROCEDE.subs(P, 0)


def marges() -> Marges:
    """Les marges de la boucle corrigee."""
    return marges_de_stabilite(OUVERTE)


def caracteristiques() -> CaracteristiquesDeM:
    """`|M(0)|`, `|M|max` et la bande passante de la boucle corrigee."""
    return caracteristiques_de_m(OUVERTE)


def estimation() -> Estimation:
    """La predetermination de la reponse indicielle par les regles du cours."""
    m = caracteristiques()
    return estimer(
        en_continu_db=m.en_continu_en_decibels(),
        maximum_db=max(m.maximum_en_decibels(), m.en_continu_en_decibels()),
        bande=m.bande_passante,
    )


def reponse_a_la_consigne() -> bo.Reponse:
    """La reponse indicielle simulee de la boucle corrigee."""
    instants, sortie = bo.reponse_indicielle(OUVERTE, DUREE)
    return bo.caracteriser(instants, sortie, 1.0)


def reponse_a_la_perturbation() -> tuple[np.ndarray, np.ndarray]:
    """L'ecart de la sortie sous un echelon unite de `x_B`, consigne inchangee."""
    aval = bo.MELANGE * bo.VANNE
    cuves = bo.PROCEDE / aval
    transfert = sp.cancel(PERTURBATION * cuves / (1 + OUVERTE))
    numerateur, denominateur = sp.fraction(transfert)
    coefficients = tuple(
        [float(c) for c in sp.Poly(e, P).all_coeffs()]
        for e in (numerateur, denominateur)
    )
    instants = np.linspace(0.0, 4 * DUREE, 4001)
    _, ecart = signal.step(coefficients, T=instants)
    return instants, np.asarray(ecart, dtype=float)


def marge_de_phase_par_scipy() -> float:
    """La marge de phase, relue sur `scipy.signal.bode`."""
    numerateur, denominateur = sp.fraction(OUVERTE)
    coefficients = tuple(
        [float(c) for c in sp.Poly(e, P).all_coeffs()]
        for e in (numerateur, denominateur)
    )
    pulsations = np.logspace(-4, -1, 30001)
    _, gains, phases = signal.bode(coefficients, pulsations)
    rang = int(np.argmin(np.abs(gains)))
    return float(180 + math.fmod(phases[rang], 360))
