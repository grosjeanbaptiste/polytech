"""La transformee de Fourier d'une porte, predite puis estimee.

Le TP fait predire `T rect(t/T) -> T sinc(T f)` par le changement
d'echelle, puis la retrouver avec `freqz` sur la porte echantillonnee.
`freqz` rend la transformee **discrete** : il faut la multiplier par la
periode d'echantillonnage pour comparer a la transformee continue.

Le piege du TP est aux bords : `rect` vaut un demi en `|t| = T/2`.
Echantillonner les deux bords a un compte un echantillon de trop, et la
valeur en zero devient `T + 1/f_e` au lieu de `T`.
"""

from __future__ import annotations

import numpy as np
import pytest

from polytech.signaux.domain.spectre_estime import (
    porte_echantillonnee,
    spectre_estime,
    transformee_de_la_porte,
)


def test_la_porte_de_dix_secondes_a_dix_hertz_porte_cent_echantillons_utiles() -> None:
    """Cent uns, plus deux demis aux bords : la somme vaut cent."""
    echantillons = porte_echantillonnee(10.0, 10.0, zeros_ajoutes=0)
    assert echantillons.sum() == pytest.approx(100.0)
    assert echantillons[0] == 0.5


def test_les_zeros_ajoutes_allongent_le_signal() -> None:
    assert len(porte_echantillonnee(10.0, 10.0, zeros_ajoutes=1000)) == 1101


def test_la_transformee_de_la_porte_vaut_sa_duree_en_zero() -> None:
    assert transformee_de_la_porte(10.0, np.array([0.0]))[0] == pytest.approx(10.0)


def test_la_transformee_de_la_porte_s_annule_a_l_inverse_de_sa_duree() -> None:
    assert transformee_de_la_porte(10.0, np.array([0.1]))[0] == pytest.approx(0.0)


def test_l_estimation_rejoint_la_prediction() -> None:
    """Sur les premiers lobes, l'ecart reste sous un pour cent de la crete."""
    frequences = np.linspace(0.0, 0.5, 201)
    estime = spectre_estime(
        porte_echantillonnee(10.0, 10.0, zeros_ajoutes=1000), 10.0, frequences
    )
    predit = transformee_de_la_porte(10.0, frequences)
    assert np.max(np.abs(estime - predit)) < 0.1


def test_une_frequence_d_echantillonnage_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="echantillonnage"):
        porte_echantillonnee(10.0, 0.0, zeros_ajoutes=0)
