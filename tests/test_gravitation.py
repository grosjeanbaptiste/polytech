"""La gravitation et le rayon de Bohr : constantes uniques, force de Newton."""

from __future__ import annotations

import pytest

from polytech.chemistry.domain.atomes import RAYON_DE_BOHR
from polytech.mecanique.domain.gravitation import (
    CONSTANTE_GRAVITATIONNELLE,
    force_gravitationnelle,
)


def test_la_force_est_newtonienne() -> None:
    assert force_gravitationnelle(1, 1, 1) == CONSTANTE_GRAVITATIONNELLE
    assert force_gravitationnelle(2, 3, 1) == pytest.approx(
        6 * CONSTANTE_GRAVITATIONNELLE, rel=1e-15
    )
    assert force_gravitationnelle(1, 1, 2) == pytest.approx(
        CONSTANTE_GRAVITATIONNELLE / 4, rel=1e-15
    )


@pytest.mark.parametrize(("m1", "m2", "r"), [(-1, 1, 1), (1, -1, 1), (1, 1, 0)])
def test_une_masse_negative_ou_une_distance_nulle_leve(
    m1: float, m2: float, r: float
) -> None:
    with pytest.raises(ValueError, match=r"masse|distance"):
        force_gravitationnelle(m1, m2, r)


def test_les_valeurs_sont_celles_de_codata() -> None:
    assert CONSTANTE_GRAVITATIONNELLE == 6.6743e-11
    assert RAYON_DE_BOHR == 5.29177210903e-11
