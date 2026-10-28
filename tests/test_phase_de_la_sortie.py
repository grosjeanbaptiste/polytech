"""La phase d'une sortie : celle de l'entree plus l'argument de `H`, en degres."""

from __future__ import annotations

import math

import pytest

from polytech.electronique.domain.phaseurs import phase_de_la_sortie


def test_les_phases_s_ajoutent() -> None:
    assert phase_de_la_sortie(30.0, -170.0) == pytest.approx(-140.0, abs=1e-9)


def test_soixante_degres_sont_pi_sur_trois() -> None:
    """Lab 1, p. 15 : l'officiel prend `pi/6` pour 60 degres, qui est `pi/3`."""
    soixante = math.degrees(math.pi / 3)
    assert phase_de_la_sortie(soixante, -170.0) == pytest.approx(-110.0, abs=1e-9)


def test_la_phase_est_ramenee_dans_un_tour() -> None:
    assert phase_de_la_sortie(170.0, 20.0) == pytest.approx(-170.0, abs=1e-9)


def test_moins_cent_quatre_vingts_reste_la_borne_haute() -> None:
    assert phase_de_la_sortie(90.0, 90.0) == pytest.approx(180.0, abs=1e-9)
