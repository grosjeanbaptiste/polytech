"""La TFD au-dela de la definition : zeros ajoutes, `NFFT` points, recentrage.

Syllabus, section 10.12.2 : un signal de `L` echantillons se complete par des
zeros jusqu'a `NFFT = 2^nextpow2(L)` et son spectre vaut `fft(y, NFFT) / L` —
la normalisation reste celle du signal d'origine, pas celle du signal
complete. Le recentrage place la frequence nulle au milieu du spectre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from polytech.maths.domain.fourier.discrete import _noyau

if TYPE_CHECKING:
    from numpy.typing import NDArray


def puissance_de_deux_suivante(nombre: int) -> int:
    """La plus petite puissance de deux qui vaut au moins `nombre`."""
    if nombre < 1:
        msg = f"nombre >= 1 requis, recu {nombre}"
        raise ValueError(msg)
    return 1 << (nombre - 1).bit_length()


def completer_par_des_zeros(
    echantillons: NDArray[np.float64], nombre: int
) -> NDArray[np.float64]:
    """Le signal suivi de zeros jusqu'a `nombre` points."""
    manque = nombre - len(echantillons)
    if manque < 0:
        msg = f"{nombre} points : moins de points que le signal ({len(echantillons)})"
        raise ValueError(msg)
    return np.concatenate([echantillons, np.zeros(manque)])


def spectre_a_nfft_points(
    echantillons: NDArray[np.float64], nfft: int
) -> NDArray[np.complex128]:
    """`X[k] = (1/L) somme_n x[n] e^(-j 2 pi k n / NFFT)`, `NFFT` coefficients."""
    complet = completer_par_des_zeros(echantillons, nfft)
    spectre: NDArray[np.complex128] = _noyau(nfft, -1) @ complet / len(echantillons)
    return spectre


def recentrer(spectre: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """La frequence nulle au milieu, sur chaque axe (une ou deux dimensions)."""
    decalage = tuple(n // 2 for n in spectre.shape)
    centre: NDArray[np.complex128] = np.roll(
        spectre, decalage, axis=tuple(range(spectre.ndim))
    )
    return centre
