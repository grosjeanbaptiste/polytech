"""L'approximation de Boussinesq : rho varie avec T dans le seul terme de flottabilite.

Oracles : le gaz parfait (`rho = rho_0 T_0 / T`, beta = 1 / T), et l'eau
de 20 a 30 degres C (998,21 a 995,65 kg/m^3, tables usuelles, beta =
2,07e-4 K^-1).
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.boussinesq import (
    boussinesq_est_licite,
    flottabilite_volumique,
    masse_volumique_de_boussinesq,
)
from polytech.fluides.domain.grandeurs import PESANTEUR
from polytech.fluides.domain.proprietes import coefficient_de_dilatation_du_gaz_parfait


def test_sans_ecart_de_temperature_la_masse_volumique_est_celle_de_reference() -> None:
    assert masse_volumique_de_boussinesq(998.21, 2.07e-4, 293.15, 293.15) == 998.21


def test_l_eau_chauffee_de_dix_degres_perd_un_quart_de_pour_cent() -> None:
    rho = masse_volumique_de_boussinesq(998.21, 2.07e-4, 303.15, 293.15)
    assert rho == pytest.approx(995.65, abs=0.6)  # beta croit avec T : ecart de 0,05 %


def test_un_fluide_plus_chaud_est_plus_leger() -> None:
    assert masse_volumique_de_boussinesq(1.2, 3e-3, 310.0, 300.0) < 1.2


def test_un_fluide_plus_froid_est_plus_lourd() -> None:
    assert masse_volumique_de_boussinesq(1.2, 3e-3, 290.0, 300.0) > 1.2


def test_la_flottabilite_vaut_rho_g_beta_delta_t() -> None:
    attendu = 1.2 * PESANTEUR * 3e-3 * 5.0
    assert flottabilite_volumique(1.2, 3e-3, 305.0, 300.0) == pytest.approx(attendu)


def test_la_flottabilite_est_negative_pour_un_fluide_plus_froid() -> None:
    assert flottabilite_volumique(1.2, 3e-3, 295.0, 300.0) < 0


def test_la_flottabilite_est_la_poussee_d_archimede_nette_du_gaz_parfait() -> None:
    t0, ecart = 300.0, 3.0
    exacte = 1.2 * (1 - t0 / (t0 + ecart)) * PESANTEUR
    modele = flottabilite_volumique(
        1.2, coefficient_de_dilatation_du_gaz_parfait(t0), t0 + ecart, t0
    )
    assert modele == pytest.approx(exacte, rel=1e-2)


def test_l_ecart_au_gaz_parfait_exact_croit_avec_l_ecart_de_temperature() -> None:
    def erreur(ecart: float) -> float:
        t0 = 300.0
        exacte = 1.2 * (1 - t0 / (t0 + ecart))
        beta = coefficient_de_dilatation_du_gaz_parfait(t0)
        return abs(exacte - 1.2 * beta * ecart) / exacte

    assert erreur(30.0) > erreur(3.0)


def test_l_approximation_est_licite_quand_beta_delta_t_est_petit() -> None:
    assert boussinesq_est_licite(2.07e-4, 10.0, tolerance=0.01)
    assert not boussinesq_est_licite(3.3e-3, 100.0, tolerance=0.01)


def test_le_critere_est_inclusif_a_la_tolerance() -> None:
    assert boussinesq_est_licite(1e-3, 10.0, tolerance=1e-2)


@pytest.mark.parametrize(
    ("beta", "tolerance"), [(0.0, 0.01), (-1e-3, 0.01), (1e-3, 0.0), (1e-3, -0.1)]
)
def test_un_coefficient_ou_une_tolerance_non_positifs_levent(
    beta: float, tolerance: float
) -> None:
    with pytest.raises(ValueError, match=r"coefficient|tolerance"):
        boussinesq_est_licite(beta, 10.0, tolerance=tolerance)


def test_une_masse_volumique_de_reference_non_positive_leve() -> None:
    with pytest.raises(ValueError, match="masse volumique"):
        masse_volumique_de_boussinesq(0.0, 3e-3, 310.0, 300.0)


def test_la_loi_lineaire_qui_annule_la_masse_volumique_leve() -> None:
    with pytest.raises(ValueError, match="Boussinesq"):
        masse_volumique_de_boussinesq(1.0, 1e-2, 500.0, 300.0)
