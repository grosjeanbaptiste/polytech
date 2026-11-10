"""La TFD : complement a zeros, spectre a `NFFT` points, recentrage.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
section 10.12.2 : `NFFT = 2^nextpow2(L)`, `Y = fft(y, NFFT) / L`, deux
sinusoides de 50 et 120 Hz echantillonnees a 1000 Hz sur `L = 1000` points.
Chemin independant : `numpy.fft`, que le module n'emploie pas.
"""

import numpy as np
import pytest

from polytech.maths.domain.fourier.discrete_etendue import (
    completer_par_des_zeros,
    puissance_de_deux_suivante,
    recentrer,
    spectre_a_nfft_points,
)

CADENCE = 1000.0
LONGUEUR = 1000
TEMPS = np.arange(LONGUEUR) / CADENCE
SIGNAL = 0.7 * np.sin(2 * np.pi * 50 * TEMPS) + np.sin(2 * np.pi * 120 * TEMPS)


@pytest.mark.parametrize(
    ("n", "attendu"), [(1, 1), (2, 2), (3, 4), (1000, 1024), (1024, 1024), (1025, 2048)]
)
def test_la_puissance_de_deux_suivante(n: int, attendu: int) -> None:
    assert puissance_de_deux_suivante(n) == attendu


def test_le_complement_a_zeros_garde_le_signal_et_ajoute_des_zeros() -> None:
    complete = completer_par_des_zeros(np.array([1.0, 2.0, 3.0]), 5)
    assert complete.tolist() == [1.0, 2.0, 3.0, 0.0, 0.0]


def test_completer_a_moins_que_la_longueur_est_refuse() -> None:
    with pytest.raises(ValueError, match="moins de points que le signal"):
        completer_par_des_zeros(np.array([1.0, 2.0, 3.0]), 2)


def test_le_spectre_a_nfft_points_est_celui_de_numpy_divise_par_la_longueur() -> None:
    nfft = puissance_de_deux_suivante(LONGUEUR)
    obtenu = spectre_a_nfft_points(SIGNAL, nfft)
    attendu = np.fft.fft(SIGNAL, nfft) / LONGUEUR
    assert np.allclose(obtenu, attendu, atol=1e-9)


def test_le_spectre_garde_les_deux_raies_de_50_et_120_hz() -> None:
    nfft = puissance_de_deux_suivante(LONGUEUR)
    amplitudes = 2 * np.abs(spectre_a_nfft_points(SIGNAL, nfft))[: nfft // 2 + 1]
    frequences = CADENCE / 2 * np.linspace(0, 1, nfft // 2 + 1)
    plus_fortes = sorted(frequences[np.argsort(amplitudes)[-2:]])
    assert plus_fortes[0] == pytest.approx(50, abs=1)
    assert plus_fortes[1] == pytest.approx(120, abs=1)


@pytest.mark.parametrize("nombre", [8, 9])
def test_recentrer_met_la_frequence_nulle_au_milieu(nombre: int) -> None:
    spectre = np.arange(nombre, dtype=np.complex128)
    centre = recentrer(spectre)
    assert centre[nombre // 2] == spectre[0]
    assert np.allclose(centre, np.fft.fftshift(spectre))


def test_recentrer_en_deux_dimensions_decale_les_deux_axes() -> None:
    spectre = np.arange(20, dtype=np.complex128).reshape(4, 5)
    centre = recentrer(spectre)
    assert centre[2, 2] == spectre[0, 0]
    assert np.allclose(centre, np.fft.fftshift(spectre))


def test_completer_a_la_meme_longueur_ne_change_rien() -> None:
    signal = np.array([1.0, 2.0, 3.0])
    assert completer_par_des_zeros(signal, 3).tolist() == [1.0, 2.0, 3.0]
