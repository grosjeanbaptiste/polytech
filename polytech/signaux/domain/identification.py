"""L'identification d'un systeme sur sa reponse indicielle.

Syllabus de signaux et systemes, sections 5.1.3 et 5.2.3. La reponse
indicielle se mesure plus facilement que l'impulsionnelle, et l'on en
tire les parametres du modele.

Premier ordre : `K - a(t) = K exp(-t / T)`, donc `ln(K - a(t))` est une
droite, d'ordonnee a l'origine `ln K` et de pente `-1 / T`.

Second ordre a poles complexes : l'instant du maximum et le taux de
depassement donnent `xi` et `T`, par `t_max = pi T / racine(1 - xi^2)`
et `D = exp(-xi pi / racine(1 - xi^2))`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def parametres_depuis_la_droite(
    *, pente: float, ordonnee: float
) -> tuple[float, float]:
    """`(K, T)` : `K = exp(ordonnee)`, `T = -1 / pente`."""
    if pente >= 0:
        msg = f"pente nulle ou positive : {pente} -- la reponse ne converge pas"
        raise ValueError(msg)
    return (math.exp(ordonnee), -1 / pente)


def identifier_le_premier_ordre(
    instants: NDArray[np.float64], reponse: NDArray[np.float64], *, regime: float
) -> tuple[float, float]:
    """`(K, T)` par la droite des moindres carres de `ln(regime - reponse)`."""
    ecart = regime - reponse
    if np.any(ecart <= 0):
        msg = "regime deja atteint ou depasse : le logarithme n'est pas defini"
        raise ValueError(msg)
    pente, ordonnee = np.polyfit(instants, np.log(ecart), 1)
    return parametres_depuis_la_droite(pente=float(pente), ordonnee=float(ordonnee))


def identifier_le_second_ordre(
    *, instant_du_maximum: float, depassement: float
) -> tuple[float, float]:
    """`(xi, T)` depuis l'instant du maximum et le depassement, en fraction."""
    if not 0 < depassement < 1:
        msg = f"depassement hors de ]0, 1[ : {depassement}"
        raise ValueError(msg)
    logarithme = math.log(depassement)
    amortissement = -logarithme / math.hypot(math.pi, logarithme)
    constante = instant_du_maximum * math.sqrt(1 - amortissement**2) / math.pi
    return (amortissement, constante)
