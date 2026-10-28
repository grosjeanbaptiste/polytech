"""Coefficient global, lame d'air et echangeurs de la fiche, par `polytech/`."""

from __future__ import annotations

import math

from polytech.maths.domain.comparaison import proches, proches_relativement
from polytech.thermique.domain.coefficient_global import (
    coefficient_global_plan,
    temperature_de_paroi,
)
from polytech.thermique.domain.conduction.parois import (
    Couche,
    Tube,
    flux_entre_deux_fluides,
    flux_lineique_de_tubes,
    temperatures_d_interface,
)
from polytech.thermique.domain.double_vitrage import (
    Vitrage,
    coefficient_de_la_lame,
    coefficient_de_rayonnement_entre_vitres,
)
from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.vitre_selective import (
    coefficient_entre_plans_gris,
)

SIGMA_T = emittance(1.0)
VITRAGE = Vitrage(epaisseur=0.003, conductivite=0.7, echanges=(23.0, 8.0))


def test_la_paroi_plane_somme_les_resistances_par_unite_d_aire() -> None:
    """`1/K_gl = 1/K_i + somme e/lambda + 1/K_e`."""
    ki, ke, couches = 15.0, 8.0, [Couche(0.2, 0.9), Couche(0.05, 0.04)]
    fiche = 1 / (1 / ki + 0.2 / 0.9 + 0.05 / 0.04 + 1 / ke)
    flux = flux_entre_deux_fluides(couches, interieur=(ki, 20.0), exterieur=(ke, 0.0))
    assert proches_relativement(flux, fiche * 20.0, 1e-12)
    assert proches_relativement(
        coefficient_global_plan(ki, 0.2, 0.9, ke),
        1 / (1 / ki + 0.2 / 0.9 + 1 / ke),
        1e-12,
    )


def test_le_tube_rapporte_chaque_resistance_a_sa_surface() -> None:
    """`1/(K_gl S) = 1/(K_i S_i) + ln(r_e/r_i)/(2 pi lambda L) + 1/(K_e S_e)`."""
    ri, re, lam, ki, ke = 0.02, 0.03, 50.0, 200.0, 10.0
    resistance = (
        1 / (ki * 2 * math.pi * ri)
        + math.log(re / ri) / (2 * math.pi * lam)
        + 1 / (ke * 2 * math.pi * re)
    )
    flux = flux_lineique_de_tubes(
        [Tube(ri, re, lam)], 100.0, 20.0, interieur=ki, exterieur=ke
    )
    assert proches_relativement(flux, 80.0 / resistance, 1e-12)


def test_la_temperature_de_paroi_retranche_la_chute_et_les_chutes_suivent_r() -> None:
    """`t_p = t_f - Phi/(K S)` ; `dt_k / dt_total = R_k / somme R`."""
    flux_surfacique = 450.0
    assert proches(
        temperature_de_paroi(90.0, flux_surfacique, 15.0), 90.0 - 450.0 / 15.0, 1e-12
    )
    couches = [Couche(0.1, 0.5), Couche(0.05, 0.1), Couche(0.02, 1.0)]
    interfaces, _ = temperatures_d_interface(couches, 100.0, 20.0)
    r = [c.resistance() for c in couches]
    assert proches_relativement(100.0 - interfaces[0], 80.0 * r[0] / sum(r), 1e-12)
    assert proches_relativement(
        interfaces[0] - interfaces[1], 80.0 * r[1] / sum(r), 1e-12
    )


def test_la_lame_d_air_ajoute_conduction_de_lame_et_rayonnement() -> None:
    """`K_lame = K_c + K_r`, `K_r = 4 sigma T_m^3 / (2/eps - 1)` ; vitrages 6 et 3,3."""
    k_c = coefficient_de_la_lame(0.006, 0.023, 980.0 * 0.71)
    k_r = coefficient_de_rayonnement_entre_vitres(0.85, 283.0)
    assert proches_relativement(k_r, 4 * SIGMA_T * 283.0**3 / (2 / 0.85 - 1), 1e-12)
    assert proches_relativement(
        k_r, coefficient_entre_plans_gris(0.85, 0.85, 283.001, 282.999), 1e-6
    )
    assert proches(k_c, 3.83, 0.01)
    assert proches(k_r, 3.8, 0.01)
    assert proches(k_r / k_c, 1.0, 0.05)  # le rayonnement ne domine pas : il egale
    assert round(VITRAGE.coefficient_global(())) == 6
    assert proches(VITRAGE.coefficient_global((k_c + k_r,)), 3.3, 0.06)
