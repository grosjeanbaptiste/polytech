"""La paroi exterieure d'un toit, entre l'interieur froid et le soleil.

Bilan sur la paroi : ce qui part vers l'interieur, plus ce que le soleil
apporte, plus ce que l'air et l'environnement echangent, fait zero. Le
rayonnement en `T^4` le rend non lineaire ; la bibliotheque le resout tel
quel, sans le lineariser par un coefficient `K_ray` qu'il faudrait
iterer.

Oracles : sans soleil ni rayonnement, la moyenne ponderee des deux
temperatures ; et dans tous les cas, un residu de bilan nul.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.paroi_exposee import (
    ToitExpose,
)

SIGMA = 5.670374419e-8


def test_sans_soleil_ni_rayonnement_la_moyenne_ponderee() -> None:
    toit = ToitExpose(1.0, 260.0, 15.0, 300.0, 0.0, 0.0, 0.0)
    attendu = (1.0 * 260.0 + 15.0 * 300.0) / 16.0
    assert toit.temperature() == pytest.approx(attendu)


def test_le_bilan_s_annule_a_la_temperature_trouvee() -> None:
    toit = ToitExpose(1.018, 260.15, 15.0, 303.15, 0.98, 800.0, 0.91)
    t = toit.temperature()
    residu = (
        1.018 * (260.15 - t)
        + 0.98 * 800.0
        + 15.0 * (303.15 - t)
        + 0.91 * SIGMA * (303.15**4 - t**4)
    )
    assert residu == pytest.approx(0.0, abs=1e-6)


def test_la_charge_frigorifique_est_le_flux_vers_l_interieur() -> None:
    toit = ToitExpose(1.018, 260.15, 15.0, 303.15, 0.98, 800.0, 0.91)
    assert toit.flux_vers_l_interieur(10.0) == pytest.approx(
        1.018 * (toit.temperature() - 260.15) * 10.0
    )


def test_le_coefficient_de_rayonnement_linearise_redonne_le_flux() -> None:
    """`K_ray = eps sigma (T^4 - Te^4) / (T - Te)`, celui que l'officiel itere."""
    toit = ToitExpose(1.018, 260.15, 15.0, 303.15, 0.98, 800.0, 0.91)
    t = toit.temperature()
    assert toit.coefficient_de_rayonnement() * (t - 303.15) == pytest.approx(
        0.91 * SIGMA * (t**4 - 303.15**4)
    )
