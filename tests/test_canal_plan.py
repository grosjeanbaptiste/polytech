"""La convection forcee entre deux plaques paralleles, en turbulent developpe.

`Nu = 0,0158 Re^0,8` sur le diametre hydraulique, qui vaut deux fois
l'ecartement des plaques. Oracle : le canal du TP 3, ou l'officiel trouve
`Re` voisin de 5500 et `K = 22,9 W/(m2 K)`.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.convection.canal_plan import (
    coefficient_du_canal,
    diametre_hydraulique,
    reynolds_du_canal,
)


def test_le_diametre_hydraulique_vaut_deux_ecartements() -> None:
    assert diametre_hydraulique(0.01) == pytest.approx(0.02)


def test_le_reynolds_du_capteur() -> None:
    assert reynolds_du_canal(0.056, 1.0, 0.01, 2.036e-5) == pytest.approx(5501, abs=1)


def test_le_coefficient_du_capteur() -> None:
    k = coefficient_du_canal(0.056, 1.0, 0.01, 2.036e-5, 0.0295)
    assert k == pytest.approx(22.9, abs=0.05)


def test_un_ecoulement_laminaire_est_refuse() -> None:
    """La correlation est celle du turbulent : en dessous, elle ment."""
    with pytest.raises(ValueError, match="turbulent"):
        coefficient_du_canal(0.001, 1.0, 0.01, 2.036e-5, 0.0295)
