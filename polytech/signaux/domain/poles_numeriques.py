"""Poles, zeros et reponses d'une transmittance donnee par ses coefficients.

Le pendant numerique de `Transmittance`, qui est symbolique : le TP 1
fait employer `tf2zpk`, `impulse` et `step`, et ses transmittances ont des
coefficients a huit decimales qu'aucune factorisation exacte ne rendrait
lisibles.

Un pole `-s + j w` impose a la reponse un facteur `e^{-s t}` qui oscille a
`w / 2 pi` hertz. D'ou les deux grandeurs lues sur un graphique : la
constante de temps `1/s` et la frequence. Un pole a partie reelle
positive n'a **pas** de constante d'amortissement -- sa reponse diverge --
et la fonction leve plutot que de rendre une duree negative.

Les poles **dominants** sont les plus proches de l'axe imaginaire : les
plus lents a s'eteindre, donc ceux qui dessinent la reponse une fois les
autres eteints. L'approximation qui ne garde qu'eux conserve le gain
statique `H(0)`, donc la valeur finale de la reponse indicielle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TOLERANCE_DE_DOMINANCE = 1e-9
"""Deux poles conjugues ont la meme partie reelle, a l'arrondi pres."""

DEGRE_DE_ROUTH = 4
"""Quatre coefficients : le troisieme degre, seul cas que la regle couvre."""


def poles_et_zeros(
    numerateur: Sequence[float], denominateur: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Les zeros, les poles et le gain, comme `tf2zpk` les rend."""
    zeros, poles, gain = signal.tf2zpk(numerateur, denominateur)
    return np.asarray(poles), np.asarray(zeros), float(gain)


def constante_de_temps(pole: complex) -> float:
    """`1/s` pour un pole `-s + j w` ; leve si le pole est instable."""
    if pole.real >= 0:
        msg = f"pole instable {pole} : sa reponse diverge, elle ne s'amortit pas"
        raise ValueError(msg)
    return -1.0 / pole.real


def frequence_d_oscillation(pole: complex) -> float:
    """`|w| / 2 pi`, en hertz."""
    return abs(pole.imag) / (2 * np.pi)


def poles_dominants(poles: np.ndarray) -> np.ndarray:
    """Les poles de plus grande partie reelle : les plus lents a s'eteindre."""
    plus_lente = np.max(poles.real)
    retenus = np.abs(poles.real - plus_lente) < TOLERANCE_DE_DOMINANCE
    return np.asarray(poles[retenus])


def est_stable(poles: np.ndarray) -> bool:
    """Vrai si tous les poles sont a partie reelle strictement negative."""
    return bool(np.all(poles.real < 0))


def routh_du_troisieme_degre(coefficients: Sequence[float]) -> float:
    """`a2 a1 - a3 a0` : positif (coefficients positifs) si et seulement si stable."""
    if len(coefficients) != DEGRE_DE_ROUTH:
        msg = f"troisieme degre attendu, {len(coefficients) - 1} recu"
        raise ValueError(msg)
    trois, deux, un, zero = coefficients
    return deux * un - trois * zero


_POINTS_DE_LA_GRILLE = 20_001


def _simulee(
    simulateur: Callable[..., tuple[np.ndarray, np.ndarray]],
    systeme: tuple[Sequence[float], Sequence[float]],
    instants: np.ndarray,
) -> np.ndarray:
    """La reponse aux instants donnes, quels qu'ils soient.

    `scipy.signal` exige des instants regulierement espaces et simule depuis
    `T[0]` comme si c'etait `t = 0`. Hors de ce cas, on simule sur une grille
    reguliere depuis l'origine et on interpole (spline cubique).
    """
    temps = np.asarray(instants, dtype=float)
    if temps.size == 0 or temps.min() < 0:
        msg = f"instant negatif ou liste vide : {temps}"
        raise ValueError(msg)
    regulier = temps.size > 1 and np.allclose(np.diff(temps), temps[1] - temps[0])
    if temps[0] == 0 and regulier:
        return np.asarray(simulateur(systeme, T=temps)[1], dtype=float)
    grille = np.linspace(0.0, max(float(temps.max()), 1e-12), _POINTS_DE_LA_GRILLE)
    reponse = np.asarray(simulateur(systeme, T=grille)[1], dtype=float)
    return np.asarray(CubicSpline(grille, reponse)(temps), dtype=float)


def reponse_impulsionnelle(
    numerateur: Sequence[float], denominateur: Sequence[float], instants: np.ndarray
) -> np.ndarray:
    """`h(t)`, l'original de la transmittance, aux instants donnes."""
    return _simulee(signal.impulse, (numerateur, denominateur), instants)


def reponse_indicielle(
    numerateur: Sequence[float], denominateur: Sequence[float], instants: np.ndarray
) -> np.ndarray:
    """La reponse a l'echelon, aux instants donnes."""
    return _simulee(signal.step, (numerateur, denominateur), instants)


def approximation_dominante(
    numerateur: Sequence[float], denominateur: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Les seuls poles dominants, au meme gain statique `H(0)`."""
    poles, _, _ = poles_et_zeros(numerateur, denominateur)
    reduit = np.real(np.poly(poles_dominants(poles)))
    gain_statique = numerateur[-1] / denominateur[-1]
    return [float(gain_statique * reduit[-1])], [float(c) for c in reduit]
