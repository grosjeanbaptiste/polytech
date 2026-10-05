"""La transformee de Fourier discrete d'un signal echantillonne.

Un signal traite numeriquement n'est connu que par `N` echantillons
`x[n] = x(n t_0)`. En remplacant l'integrale du coefficient de Fourier par
une somme :

    X[k] = (1/N) somme_n x[n] e^(-j 2 pi k n / N),
    x[n] = somme_k X[k] e^(+j 2 pi k n / N).

C'est la serie de Fourier du signal rendu periodique, de periode `N t_0` ;
les `X[k]` sont eux-memes periodiques, de periode `N`. La transformee
rapide (FFT) calcule la meme somme, plus vite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def echantillonner(
    fonction: Callable[[float], float], pas: float, nombre: int
) -> NDArray[np.float64]:
    """`x[n] = x(n t_0)`, pour `n` de 0 a `nombre - 1`."""
    return np.array([fonction(n * pas) for n in range(nombre)], dtype=np.float64)


def _noyau(nombre: int, signe: int) -> NDArray[np.complex128]:
    """La matrice des `e^(signe . j 2 pi k n / N)`."""
    indices = np.arange(nombre)
    return np.exp(signe * 2j * np.pi * np.outer(indices, indices) / nombre)


def transformee_discrete(echantillons: NDArray[np.float64]) -> NDArray[np.complex128]:
    """`X[k]`, par la somme de la definition ; leve sur un signal vide."""
    nombre = len(echantillons)
    if nombre == 0:
        msg = "aucun echantillon : la transformee discrete n'est pas definie"
        raise ValueError(msg)
    spectre: NDArray[np.complex128] = _noyau(nombre, -1) @ echantillons / nombre
    return spectre


def transformee_discrete_inverse(
    spectre: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """`x[n] = somme X[k] e^(j 2 pi k n / N)`, pour un signal reel."""
    signal: NDArray[np.float64] = np.real(_noyau(len(spectre), 1) @ spectre)
    return signal


def spectre_unilateral(
    echantillons: NDArray[np.float64], cadence: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Les frequences de 0 a `Fs / 2` et l'amplitude `2 |X[k]|` de chacune."""
    nombre = len(echantillons)
    moitie = nombre // 2 + 1
    frequences = cadence * np.arange(moitie) / nombre
    amplitudes = 2 * np.abs(transformee_discrete(echantillons)[:moitie])
    return frequences, amplitudes


def raies_dominantes(
    echantillons: NDArray[np.float64], cadence: float, nombre: int
) -> tuple[float, ...]:
    """Les `nombre` frequences de plus grande amplitude."""
    frequences, amplitudes = spectre_unilateral(echantillons, cadence)
    plus_fortes = np.argsort(amplitudes)[-nombre:]
    return tuple(float(frequences[i]) for i in plus_fortes)
