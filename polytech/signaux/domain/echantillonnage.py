"""L'echantillonnage et la quantification d'un signal analogique.

Notes de cours de signaux et systemes, sections 1.3 et 1.4. Echantillonner,
c'est mesurer le signal tous les `T_e` : `x(n) = x(n T_e)` (1.7). La
cadence `F_e` doit depasser deux fois la plus haute frequence du signal ;
au-dela de `F_e / 2`, qui joue le miroir, une sinusoide en imite une
autre, plus basse.

Quantifier, c'est ranger les valeurs sur `2^b` niveaux espaces du pas
`q` ; l'erreur commise reste entre `-q / 2` et `q / 2` (1.8), et s'entend
comme un bruit. Un rapport de puissances se dit en decibels,
`10 log(P_s / P_b)`, soit `20 log` du rapport des valeurs efficaces (1.9).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from polytech.signaux.domain.frequence import INSTANT

if TYPE_CHECKING:
    from numpy.typing import NDArray


def echantillonner(signal: sp.Expr, *, cadence: float, nombre: int) -> list[float]:
    """Les `nombre` premiers echantillons `x(n / F_e)`, a partir de `n = 0`."""
    if cadence <= 0:
        msg = f"cadence nulle ou negative : {cadence} Hz"
        raise ValueError(msg)
    valeur = sp.lambdify(INSTANT, signal, "math")
    return [float(valeur(rang / cadence)) for rang in range(nombre)]


def frequence_minimale_d_echantillonnage(plus_haute: float) -> float:
    """Le double de la plus haute frequence du signal, que `F_e` doit depasser."""
    return 2 * plus_haute


def frequence_apparente(frequence: float, cadence: float) -> float:
    """La frequence que les echantillons montrent, repliee sous `F_e / 2`."""
    reste = math.fmod(frequence, cadence)
    return min(reste, cadence - reste)


def quantifier(
    valeurs: NDArray[np.float64], *, bits: int
) -> tuple[NDArray[np.float64], float]:
    """Le signal quantifie et le pas `q = 2 max / 2^b`, comme l'exemple 3."""
    if bits <= 0:
        msg = f"nombre de bits nul ou negatif : {bits}"
        raise ValueError(msg)
    pas = float(2 * np.max(valeurs) / 2**bits)
    return (np.floor(valeurs / pas) * pas + pas / 2, pas)


def puissance_estimee(echantillons: NDArray[np.float64]) -> float:
    """`P = (1 / N) somme |f(n)|^2`."""
    if len(echantillons) == 0:
        msg = "aucun echantillon : la puissance n'est pas definie"
        raise ValueError(msg)
    return float(np.mean(np.abs(echantillons) ** 2))


def gain_en_decibels(rapport_de_puissances: float) -> float:
    """`10 log10` d'un rapport de puissances."""
    if rapport_de_puissances <= 0:
        msg = f"rapport nul ou negatif : {rapport_de_puissances}"
        raise ValueError(msg)
    return 10 * math.log10(rapport_de_puissances)


def plage_des_codes(bits: int) -> tuple[int, int]:
    """Les entiers signes que `b` bits codent : `2^b` niveaux, de `-2^(b-1)`."""
    if bits <= 0:
        msg = f"nombre de bits nul ou negatif : {bits}"
        raise ValueError(msg)
    moitie = 2 ** (bits - 1)
    return (-moitie, moitie - 1)
