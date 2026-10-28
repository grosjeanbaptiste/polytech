"""Les ailettes et l'inertie de la fiche de thermique, refaites par `polytech/`."""

from __future__ import annotations

import math

import sympy as sp

from polytech.maths.domain.comparaison import egales, proches_relativement
from polytech.thermique.domain.conduction.ailette_droite import (
    flux_au_pied,
    flux_d_une_surface_ailetee,
    flux_sans_resistance,
    parametre_d_ailette,
    parametre_d_ailette_mince,
    profil_d_ailette,
)
from polytech.thermique.domain.conduction.ailettes import AiletteDroite
from polytech.thermique.domain.inertie import SolideIsotherme
from polytech.thermique.domain.surface_ailetee import SurfaceAiletee

K, LAM, P, S, B, D, LONG = 25.0, 200.0, 0.08, 1e-4, 2e-3, 4e-3, 0.03


def test_le_parametre_d_ailette_est_racine_de_k_p_sur_lambda_s() -> None:
    """`m = racine(K P / (lambda S))`."""
    fiche = math.sqrt(K * P / (LAM * S))
    assert proches_relativement(parametre_d_ailette(K, P, LAM, S), fiche, 1e-12)


def test_l_ailette_mince_a_pour_parametre_racine_de_2k_sur_lambda_b() -> None:
    """Ailette mince d'epaisseur `b` : `m = racine(2 K / (lambda b))`."""
    fiche = math.sqrt(2 * K / (LAM * B))
    assert proches_relativement(parametre_d_ailette_mince(K, LAM, B), fiche, 1e-12)


def test_une_ailette_aiguille_a_p_sur_s_egal_4_sur_d() -> None:
    """Cylindre de diametre `d` : `P/S = pi d / (pi d^2/4) = 4/d`."""
    perimetre, section = math.pi * D, math.pi * D**2 / 4
    assert proches_relativement(perimetre / section, 4 / D, 1e-12)
    fiche = math.sqrt(4 * K / (LAM * D))
    assert proches_relativement(
        parametre_d_ailette(K, perimetre, LAM, section), fiche, 1e-12
    )


def test_l_efficacite_est_tanh_ml_sur_ml() -> None:
    """`Omega = tanh(m l) / (m l)`, et c'est bien flux reel / flux a `t_pied`."""
    m = parametre_d_ailette(K, P, LAM, S)
    ailette = AiletteDroite(m, LONG)
    assert proches_relativement(
        ailette.efficacite(), math.tanh(m * LONG) / (m * LONG), 1e-12
    )
    reel = flux_au_pied(m, LAM * S, ecart=50.0, longueur=LONG)
    ideal = flux_sans_resistance(K, P, ecart=50.0, longueur=LONG)
    assert proches_relativement(reel / ideal, ailette.efficacite(), 1e-12)


def test_l_ecart_decroit_en_cosh_m_l_moins_x_sur_cosh_m_l() -> None:
    """`theta / theta_1 = cosh(m (l - x)) / cosh(m l)`."""
    x, m, longueur = sp.symbols("x m l", positive=True)
    fiche = sp.cosh(m * (longueur - x)) / sp.cosh(m * longueur)
    assert egales(profil_d_ailette(x, m, longueur), fiche)


def test_la_surface_ailetee_echange_par_s0_plus_omega_sa() -> None:
    """`Phi = K theta (S_0 + Omega S_a)` et `R = 1 / (K (S_0 + Omega S_a))`."""
    s0, sa, omega, theta = 0.5, 0.3, 0.8, 40.0
    flux = flux_d_une_surface_ailetee(
        K, theta, primaire=s0, secondaire=sa, efficacite=omega
    )
    assert proches_relativement(flux, K * theta * (s0 + omega * sa), 1e-12)
    surfaces = SurfaceAiletee(primaire=0.8, libre=s0, ailettes=sa)
    resistance = surfaces.resistance(K, omega) / 0.8
    assert proches_relativement(resistance, 1 / (K * (s0 + omega * sa)), 1e-12)


def test_la_constante_de_temps_du_solide_isotherme_et_sa_decroissance() -> None:
    """`tau_c = rho c V / (K S)`, `theta / theta_0 = exp(-tau / tau_c)`."""
    rho, c, volume, surface = 7800.0, 500.0, 1e-3, 0.06
    solide = SolideIsotherme.depuis(
        echange=K, surface=surface, capacite=rho * c * volume
    )
    tau_c = rho * c * volume / (K * surface)
    assert proches_relativement(float(solide.constante_de_temps()), tau_c, 1e-12)
    tau = sp.Symbol("tau", nonnegative=True)
    assert egales(solide.ecart_reduit(tau), sp.exp(-tau / tau_c))


def test_la_vitesse_de_chauffe_est_le_flux_net_sur_la_capacite() -> None:
    """`rho c V dt/dtau = Phi_net` : pente initiale `-K S theta_0 / (rho c V)`."""
    rho_c_v, ks, theta0 = 3.9e3, 1.5, 70.0
    solide = SolideIsotherme(resistance=1 / ks, capacite=rho_c_v)
    tau = sp.Symbol("tau", nonnegative=True)
    pente = sp.diff(theta0 * solide.ecart_reduit(tau), tau).subs(tau, 0)
    assert proches_relativement(float(pente), -ks * theta0 / rho_c_v, 1e-12)
