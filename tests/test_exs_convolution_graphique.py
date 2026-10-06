"""Signaux et systemes : la note manuscrite « Convolution graphique ».

Source : `SignauxSystemes/EXS/Exs_Convolution_graphique.jpg`, une page
manuscrite, transcrite dans
`transcriptions/SignauxSystemes/EXS/Exs_Convolution_graphique.md`.

La note convolue deux creneaux en faisant glisser l'un sur l'autre. Son
croquis final -- `1/2` en `t = 1`, `-1/2` en `t = 2`, nul hors de
`[0, 3]` -- sert d'oracle, et trois chemins le recoupent : l'integrale de
convolution du domaine, l'aire de recouvrement, `numpy.convolve`.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.application import convolution_graphique as cg


def test_les_sommets_du_croquis() -> None:
    """`y(1) = 1/2`, `y(2) = -1/2`, et la courbe revient a zero en `t = 3`."""
    assert cg.sortie(1) == sp.Rational(1, 2)
    assert cg.sortie(2) == -sp.Rational(1, 2)
    assert cg.sortie(3) == 0


def test_le_support_annonce_par_la_note() -> None:
    """La note : recouvrement pour `t` dans `[0, 3]`, convolution nulle ailleurs."""
    assert cg.support() == (0, 3)
    assert cg.sortie(-1) == 0
    assert cg.sortie(sp.Rational(7, 2)) == 0


def test_les_trois_morceaux() -> None:
    """`t/2`, puis `3/2 - t`, puis `(t - 3)/2` : trois segments de droite."""
    t = sp.Symbol("t")
    assert cg.morceaux(t) == (t / 2, sp.Rational(3, 2) - t, (t - 3) / 2)


def test_le_passage_par_zero() -> None:
    """Entre les deux sommets, en `t = 3/2`."""
    assert cg.sortie(sp.Rational(3, 2)) == 0


@pytest.mark.parametrize("instant", [0.25, 0.8, 1.3, 1.9, 2.4, 2.95])
def test_par_l_aire_de_recouvrement(instant: float) -> None:
    """Second chemin, celui de la note : la surface du produit, creneau par creneau."""
    assert cg.aire_du_produit(instant) == pytest.approx(float(cg.sortie(instant)))


def test_par_numpy() -> None:
    """Troisieme chemin : `numpy.convolve` sur les deux creneaux echantillonnes."""
    instants, numerique = cg.par_echantillonnage(pas=1e-3)
    exacte = np.array([float(cg.sortie(float(t))) for t in instants[::250]])
    assert np.max(np.abs(numerique[::250] - exacte)) < 2e-3


def test_l_aire_de_la_sortie_est_le_produit_des_aires() -> None:
    """L'aire de `h` est nulle (`+1` puis `-1`) : celle de `y` aussi."""
    assert cg.aire_de_la_sortie() == 0
