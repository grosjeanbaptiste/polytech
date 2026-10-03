"""Les signaux echantillonnes du TP 1 : portes, peignes, convolution.

La question « par quel `b` faut-il convoluer `a` pour obtenir `c` ? » a
deux chemins. Le premier devine `b` et verifie `a * b = c` ; le second
**deconvolue** `c` par `a` et lit `b` sans rien deviner. Les deux sont
ici, et le second leve si `c` n'est le produit de `a` par aucune suite
finie -- un reste non nul serait sinon une reponse fausse rendue en
silence.
"""

from __future__ import annotations

import numpy as np
import pytest

from polytech.signaux.domain.echantillons import convoluer, deconvoluer, peigne, porte


def test_une_porte_vaut_un_sur_sa_largeur_puis_zero() -> None:
    assert porte(3, 5).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_une_porte_plus_large_que_son_support_est_refusee() -> None:
    with pytest.raises(ValueError, match="largeur"):
        porte(6, 5)


def test_une_porte_vide_est_refusee() -> None:
    with pytest.raises(ValueError, match="largeur"):
        porte(0, 5)


def test_un_peigne_place_ses_impulsions_a_chaque_periode() -> None:
    assert np.flatnonzero(peigne(3, 4)).tolist() == [0, 3, 6, 9]


def test_un_peigne_sans_impulsion_est_refuse() -> None:
    with pytest.raises(ValueError, match="impulsion"):
        peigne(3, 0)


def test_une_porte_par_elle_meme_donne_un_triangle() -> None:
    """La convolution de deux portes de largeur `n` culmine a `n`."""
    triangle = convoluer(porte(4, 4), porte(4, 4))
    assert triangle.tolist() == [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]


def test_deconvoluer_retrouve_le_facteur() -> None:
    produit = convoluer(porte(4, 4), peigne(6, 3))
    assert np.allclose(deconvoluer(produit, porte(4, 4)), peigne(6, 3))


def test_deconvoluer_un_signal_qui_n_est_pas_un_produit_leve() -> None:
    """Un reste non nul : aucune suite finie ne convient."""
    with pytest.raises(ValueError, match="reste"):
        deconvoluer(np.array([1.0, 0.0, 0.0, 5.0]), np.array([1.0, 1.0]))
