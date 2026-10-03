"""Deux lectures d'oscilloscope sur un premier ordre : montee et deux points.

Le labo II fait mesurer la constante de temps de deux facons. Les
oracles sont ceux du cours : une exponentielle passe de 10 % a 90 % de
sa valeur finale en `tau ln 9`, et deux points de la decharge
`v e^{-t/tau}` donnent `tau = (t2 - t1) / ln(v1 / v2)`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.premier_ordre import (
    constante_par_deux_points,
    temps_de_montee,
)

TAU = sp.Symbol("tau", positive=True)
T = sp.Symbol("t", positive=True)


def test_le_temps_de_montee_vaut_tau_ln_9() -> None:
    assert sp.simplify(temps_de_montee(TAU) - TAU * sp.log(9)) == 0


def test_le_temps_de_montee_separe_bien_dix_et_quatre_vingt_dix_pour_cent() -> None:
    """Oracle par la courbe elle-meme : `1 - e^{-t/tau}` aux deux instants."""
    dix = -TAU * sp.log(sp.Rational(9, 10))
    quatre_vingt_dix = dix + temps_de_montee(TAU)
    assert sp.simplify(1 - sp.exp(-quatre_vingt_dix / TAU)) == sp.Rational(9, 10)


def test_deux_points_d_une_decharge_rendent_tau() -> None:
    tau = 1e-4
    t1, t2 = 2e-5, 1.5e-4
    v1, v2 = 5 * math.exp(-t1 / tau), 5 * math.exp(-t2 / tau)
    assert constante_par_deux_points(t1, v1, t2, v2) == pytest.approx(tau)


def test_deux_points_dans_le_mauvais_ordre_sont_refuses() -> None:
    with pytest.raises(ValueError, match="ordre"):
        constante_par_deux_points(2e-4, 1.0, 1e-4, 2.0)


def test_une_tension_qui_ne_decroit_pas_est_refusee() -> None:
    """`ln(v1/v2) <= 0` : ce n'est pas une decharge, tau serait negatif."""
    with pytest.raises(ValueError, match="decharge"):
        constante_par_deux_points(1e-4, 1.0, 2e-4, 1.0)
