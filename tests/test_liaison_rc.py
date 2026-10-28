"""Le condensateur de liaison : `C = 1 / (2 pi R f_c)` pour couper a `f_c`."""

from __future__ import annotations

import math

import pytest

from polytech.electronique.domain.liaison_rc import capacite_de_liaison


def test_seize_ohms_a_vingt_hertz_donnent_497_microfarads() -> None:
    """Lab 5 : sur 16 ohm, a 20 Hz (l'enonce annonce 995 uF)."""
    assert capacite_de_liaison(16.0, 20.0) == pytest.approx(497.36e-6, abs=0.01e-6)


def test_huit_ohms_donnent_le_double() -> None:
    assert capacite_de_liaison(8.0, 20.0) == pytest.approx(
        2 * capacite_de_liaison(16.0, 20.0), rel=1e-12
    )


def test_la_coupure_du_condensateur_retombe_sur_la_frequence() -> None:
    capacite = capacite_de_liaison(16.0, 20.0)
    assert 1 / (2 * math.pi * 16.0 * capacite) == pytest.approx(20.0, rel=1e-12)


@pytest.mark.parametrize(
    ("resistance", "frequence"), [(0.0, 20.0), (16.0, 0.0), (-1.0, 20.0)]
)
def test_une_resistance_ou_une_frequence_non_positive_leve(
    resistance: float, frequence: float
) -> None:
    with pytest.raises(ValueError, match="non positive"):
        capacite_de_liaison(resistance, frequence)
