"""TP « Reservoir », deuxieme partie : la boucle fermee par un gain `K`.

La boucle ouverte est `K x 0,009 x 794/((1 + 298p)(1 + 443p)(1 + 145p))`,
avec les coefficients arrondis que l'enonce affiche.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import sympy as sp
from scipy import signal

from polytech.signaux.domain.marges import Marges, marges_de_stabilite
from polytech.signaux.domain.stabilite import Stabilite
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.temps_caracteristiques import temps_de_montee
from polytech.signaux.domain.transmittance import Transmittance

VANNE: Final = sp.Rational(9, 1000)
MELANGE: Final = sp.Integer(794)
CONSTANTES: Final = (298, 443, 145)
"""Les trois constantes de temps du modele linearise, en secondes."""
PROCEDE: Final = VANNE * MELANGE / sp.Mul(*(1 + T * P for T in CONSTANTES))
GAIN: Final = sp.Symbol("K", positive=True)


@dataclass(frozen=True, slots=True)
class Reponse:
    """La valeur finale, le depassement en pour cent et le temps de montee."""

    finale: float
    depassement: float
    montee: float


def gain_statique() -> float:
    """`0,009 x 794` : le gain de la boucle en continu, pour `K = 1`."""
    return float(PROCEDE.subs(P, 0))


def marge_de_gain_analytique() -> tuple[float, float]:
    """La pulsation ou l'argument vaut -180 degres, puis `1/|G|` en ce point.

    Trois premiers ordres : la somme des arcs tangentes vaut 180 degres
    quand `w^2 = (T_1 + T_2 + T_3)/(T_1 T_2 T_3)`.
    """
    pulsation = math.sqrt(sum(CONSTANTES) / math.prod(CONSTANTES))
    module = gain_statique() / math.prod(
        math.hypot(1, pulsation * t) for t in CONSTANTES
    )
    return pulsation, 1 / module


def gain_limite_par_routh() -> float:
    """Le `K` qui annule la condition de Routh du troisieme degre."""
    etude = Stabilite(boucle_ouverte=Transmittance(expression=GAIN * PROCEDE))
    return float(etude.gain_critique(GAIN))


def marges(gain: float) -> Marges:
    """Les marges de gain et de phase, comme la commande `margin`."""
    return marges_de_stabilite(gain * PROCEDE)


def erreur_de_regime(gain: float) -> float:
    """`1/(1 + K G(0))` : l'ecart qui subsiste sous un echelon de consigne."""
    return 1 / (1 + gain * gain_statique())


def _fermee(ouverte: sp.Expr) -> tuple[list[float], list[float]]:
    numerateur, denominateur = sp.fraction(sp.cancel(ouverte / (1 + ouverte)))
    return (
        [float(c) for c in sp.Poly(numerateur, P).all_coeffs()],
        [float(c) for c in sp.Poly(denominateur, P).all_coeffs()],
    )


def reponse_indicielle(ouverte: sp.Expr, duree: float) -> tuple[np.ndarray, np.ndarray]:
    """La reponse a un echelon unite de consigne, la boucle refermee."""
    instants = np.linspace(0.0, duree, 4001)
    _, sortie = signal.step(_fermee(ouverte), T=instants)
    return instants, np.asarray(sortie, dtype=float)


def caracteriser(instants: np.ndarray, sortie: np.ndarray, finale: float) -> Reponse:
    """La valeur finale, le depassement et le temps de montee d'une reponse."""
    depassement = max(0.0, 100 * (float(sortie.max()) / finale - 1))
    return Reponse(
        finale, depassement, temps_de_montee(instants, sortie, regime=finale)
    )


def reponse(gain: float) -> Reponse:
    """Les caracteristiques de la reponse indicielle pour ce gain."""
    instants, sortie = reponse_indicielle(gain * PROCEDE, 20000.0)
    return caracteriser(instants, sortie, 1 - erreur_de_regime(gain))


def poles_en_boucle_fermee(gain: float) -> np.ndarray:
    """Les racines du denominateur de la boucle fermee."""
    _, denominateur = _fermee(gain * PROCEDE)
    return np.roots(denominateur)
