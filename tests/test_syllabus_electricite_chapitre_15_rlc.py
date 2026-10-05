"""Syllabus d'electricite generale, chapitre 15 : decharge RLC, parasites, diffusion.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 15-22 a 15-34, lues sur les pages rendues.

Les trois regimes de la decharge sont eprouves par leur equation
differentielle, leurs conditions initiales et leur transformee de
Laplace. Deux lectures de figure servent d'oracle : le minimum de la
figure 15.12.3 et le point `t / T = 9` de la figure 15.13.2.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.decharge_rlc import (
    Regime,
    charge_de_decharge,
    courant_de_decharge,
    regime_de_la_decharge,
    transformee_de_la_charge,
    transformee_du_courant,
)
from polytech.electromagnetisme.domain.circuits.parasites import (
    courant_rc_avec_inductance_parasite,
)
from polytech.electromagnetisme.domain.circuits.transitoires import (
    amortissement_rlc,
    pulsation_propre,
)
from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.diffusion import (
    residu_de_diffusion_cylindrique,
    residu_de_diffusion_plane,
)
from polytech.electromagnetisme.domain.effet_pelliculaire import (
    CONDUCTIVITE_DU_CUIVRE,
    epaisseur_de_peau,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, RAYON, TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import (
    VARIABLE,
    valeur_initiale,
)

Q0, T = sp.symbols("Q_0 T", positive=True)
AMORTISSEMENTS = (sp.Rational(5, 4), sp.Integer(1), sp.Rational(1, 2))
"""Un cas par regime : aperiodique, critique, pseudo-periodique."""


def test_les_trois_regimes_se_reconnaissent_a_l_amortissement() -> None:
    """Page 15-23 : le signe du discriminant `4 T^2 (xi^2 - 1)`."""
    assert regime_de_la_decharge(1.25) is Regime.APERIODIQUE
    assert regime_de_la_decharge(1) is Regime.CRITIQUE
    assert regime_de_la_decharge(0.5) is Regime.PSEUDO_PERIODIQUE
    assert regime_de_la_decharge(0) is Regime.PSEUDO_PERIODIQUE


def test_un_amortissement_negatif_n_est_pas_un_circuit_passif() -> None:
    with pytest.raises(ValueError, match="amortissement negatif"):
        regime_de_la_decharge(-0.1)


def test_la_resistance_critique_est_deux_racines_de_l_sur_c() -> None:
    """Page 15-24 : `xi = 1` pour `R = 2 racine(L / C)`."""
    assert amortissement_rlc(sp.Integer(3), sp.Integer(9), sp.Integer(4)) == 1


@pytest.mark.parametrize("xi", AMORTISSEMENTS)
def test_la_charge_satisfait_l_equation_du_circuit(xi: sp.Expr) -> None:
    """(15.12.2) en variables reduites : `T^2 q'' + 2 xi T q' + q = 0`."""
    charge = charge_de_decharge(xi, T, Q0)
    residu = T**2 * sp.diff(charge, TEMPS, 2) + 2 * xi * T * sp.diff(charge, TEMPS)
    assert sp.simplify(residu + charge) == 0


@pytest.mark.parametrize("xi", AMORTISSEMENTS)
def test_la_decharge_part_de_q0_sans_courant(xi: sp.Expr) -> None:
    """Page 15-23 : `q(0) = Q_0` par (15.8.1), `i(0) = 0` par (15.8.2)."""
    assert charge_de_decharge(xi, T, Q0).subs(TEMPS, 0) == Q0
    assert courant_de_decharge(xi, T, Q0).subs(TEMPS, 0) == 0


@pytest.mark.parametrize("xi", AMORTISSEMENTS)
def test_le_courant_est_la_derivee_de_la_charge(xi: sp.Expr) -> None:
    """(15.12.9), (15.12.13) et (15.12.17) contre `i = dq/dt` (15.12.1)."""
    derivee = sp.diff(charge_de_decharge(xi, T, Q0), TEMPS)
    assert sp.simplify(courant_de_decharge(xi, T, Q0) - derivee) == 0


def test_les_originaux_ecrits_par_le_syllabus() -> None:
    """(15.12.9) pour `xi = 5/4`, (15.12.13) pour `xi = 1`, (15.12.17) pour `1/2`."""
    reduit = TEMPS / T
    aperiodique = -(Q0 / T) * sp.sinh(sp.Rational(3, 4) * reduit) / sp.Rational(3, 4)
    aperiodique *= sp.exp(-sp.Rational(5, 4) * reduit)
    ecart = courant_de_decharge(sp.Rational(5, 4), T, Q0) - aperiodique
    assert sp.simplify(ecart.rewrite(sp.exp)) == 0
    critique = -(Q0 / T) * reduit * sp.exp(-reduit)
    assert sp.simplify(courant_de_decharge(1, T, Q0) - critique) == 0
    racine = sp.sqrt(3) / 2
    pseudo = -(Q0 / T) * sp.sin(racine * reduit) / racine * sp.exp(-reduit / 2)
    assert sp.simplify(courant_de_decharge(sp.Rational(1, 2), T, Q0) - pseudo) == 0


@pytest.mark.parametrize("xi", AMORTISSEMENTS)
def test_la_transformee_de_la_charge_est_celle_de_son_original(xi: sp.Expr) -> None:
    """(15.12.4), integree numeriquement pour `T = 1`, `Q_0 = 1`, `p = 0,7`."""
    charge = sp.lambdify(TEMPS, charge_de_decharge(xi, 1, 1).rewrite(sp.exp))
    pas, borne = 1e-3, 80.0
    instants = [pas * (k + 0.5) for k in range(int(borne / pas))]
    integrale = sum(charge(t) * math.exp(-0.7 * t) for t in instants) * pas
    attendue = transformee_de_la_charge(xi, 1, 1).subs(VARIABLE, 0.7)
    assert integrale == pytest.approx(float(attendue), rel=1e-5)


def test_la_transformee_du_courant_suit_de_celle_de_la_charge() -> None:
    """(15.12.5) : `I(p) = p Q(p) - Q_0 = -Q_0 / (p^2 T^2 + 2 xi p T + 1)`."""
    xi = sp.Symbol("xi", positive=True)
    charge = transformee_de_la_charge(xi, T, Q0)
    courant = transformee_du_courant(xi, T, Q0)
    assert sp.simplify(VARIABLE * charge - Q0 - courant) == 0
    assert valeur_initiale(charge) == Q0
    assert valeur_initiale(courant) == 0


def test_le_double_signe_de_15_12_7_contredit_15_12_5() -> None:
    """(15.12.7) imprime `I(p) = -(-Q_0 / T^2) / ((p - p_1)(p - p_2))`.

    Factorise, le denominateur de (15.12.5) vaut `T^2 (p - p_1)(p - p_2)` :
    il ne reste qu'un signe moins, celui du numerateur.
    """
    xi = sp.Rational(5, 4)
    premiere, seconde = ((-xi + s * sp.Rational(3, 4)) / T for s in (1, -1))
    facteurs = (VARIABLE - premiere) * (VARIABLE - seconde)
    courant = transformee_du_courant(xi, T, Q0)
    assert sp.simplify(courant - (-Q0 / T**2) / facteurs) == 0
    assert sp.simplify(courant - (-(-Q0 / T**2) / facteurs)) != 0


def test_le_bilan_energetique_de_la_decharge() -> None:
    """Page 15-27 : `d/dt (L i^2 / 2 + q^2 / (2 C)) + R i^2 = 0`."""
    inductance, capacite = sp.Integer(9), sp.Integer(4)
    resistance = sp.Integer(2)
    periode = sp.sqrt(inductance * capacite)
    xi = amortissement_rlc(resistance, inductance, capacite)
    charge = charge_de_decharge(xi, periode, Q0)
    courant = courant_de_decharge(xi, periode, Q0)
    energie = inductance * courant**2 / 2 + charge**2 / (2 * capacite)
    assert sp.simplify(sp.diff(energie, TEMPS) + resistance * courant**2) == 0


def test_sans_resistance_la_decharge_est_periodique() -> None:
    """Remarque de la page 15-26 : `xi = 0`, pulsation `omega_0 = 1 / racine(L C)`."""
    charge = charge_de_decharge(0, T, Q0)
    assert sp.simplify(charge - Q0 * sp.cos(TEMPS / T)) == 0
    propre = pulsation_propre(9, 4)
    assert propre == 1 / sp.sqrt(36)
    assert sp.simplify(charge.subs(T, 6) - Q0 * sp.cos(propre * TEMPS)) == 0


def test_le_minimum_de_la_figure_15_12_3() -> None:
    """`xi = 1` : `R C i / Q_0 = -2 (t / T) exp(-t / T)`, minimal en `t = T`.

    La tension reduite aux bornes de la resistance y vaut `-2 / e`, soit
    -0,74 : c'est le creux que la figure montre a l'abscisse 1.
    """
    reduite = 2 * T * courant_de_decharge(1, T, Q0) / Q0
    assert sp.diff(reduite, TEMPS).subs(TEMPS, T) == 0
    assert float(reduite.subs(TEMPS, T)) == pytest.approx(-0.74, abs=0.01)


def test_un_grand_amortissement_decharge_lentement() -> None:
    """Page 15-24 : plus `xi` est grand, plus la decharge est lente."""
    reste = [
        float(charge_de_decharge(xi, 1, 1).subs(TEMPS, 9).rewrite(sp.exp))
        for xi in (sp.Rational(5, 4), sp.Rational(13, 4), sp.Rational(21, 4))
    ]
    assert reste[0] < reste[1] < reste[2]
    # Figure 15.12.2 : la courbe 5,25 est encore a 0,42 en `t / T = 9`.
    assert reste[2] == pytest.approx(0.42, abs=0.01)


E, R = sp.symbols("E R", positive=True)


def test_l_inductance_parasite_empeche_le_courant_de_sauter() -> None:
    """(15.13.2) : le courant part de zero, et non de `E / R`."""
    courant = courant_rc_avec_inductance_parasite(E, R, sp.Integer(5), T)
    assert courant.subs(TEMPS, 0) == 0
    # (15.12.5) avec `Q_0 = -C E`, et `C = tau / R = 2 xi T / R`.
    capacite = 2 * 5 * T / R
    attendu = courant_de_decharge(sp.Integer(5), T, -capacite * E)
    assert sp.simplify(courant - attendu) == 0
    # A l'instant zero toute la source est aux bornes de `l = T^2 / C`.
    parasite = T**2 / capacite
    pente = sp.diff(courant, TEMPS).subs(TEMPS, 0)
    assert sp.simplify(parasite * pente - E) == 0


def test_la_figure_15_13_2_se_superpose_au_circuit_ideal() -> None:
    """`xi = 5`, donc `tau = 10 T` : 0,41 en `t / T = 9` pour les deux courbes."""
    reel = courant_rc_avec_inductance_parasite(1, 1, sp.Integer(5), 1)
    en_neuf = float(reel.subs(TEMPS, 9).rewrite(sp.exp))
    ideal = math.exp(-9 / 10)
    assert ideal == pytest.approx(0.41, abs=0.005)
    assert en_neuf == pytest.approx(ideal, rel=0.02)


def test_l_exposant_de_15_13_3_est_t_sur_deux_xi_t() -> None:
    """(15.13.3) imprime `exp(-2 xi t / T)`, soit `exp(-90)` en `t / T = 9`.

    `tau = 2 xi T` : l'exposant est `-t / (2 xi T)`, et c'est lui que la
    figure trace.
    """
    reel = courant_rc_avec_inductance_parasite(1, 1, sp.Integer(5), 1)
    en_neuf = float(reel.subs(TEMPS, 9).rewrite(sp.exp))
    assert en_neuf > 1e30 * math.exp(-2 * 5 * 9)


def test_un_circuit_rc_peu_amorti_n_est_pas_le_cas_de_la_figure() -> None:
    with pytest.raises(ValueError, match="amortissement"):
        courant_rc_avec_inductance_parasite(E, R, sp.Rational(1, 2), T)


def test_le_courant_continu_uniforme_satisfait_la_diffusion() -> None:
    """(15.14.2) : second membre nul, la repartition est uniforme."""
    assert residu_de_diffusion_plane(sp.Integer(3), 1, 1) == 0
    assert residu_de_diffusion_cylindrique(sp.Integer(3), 1, 1) == 0


def test_un_courant_variable_n_est_pas_uniforme() -> None:
    """Page 15-31 : sinon le premier membre de (15.14.3) serait nul et pas le second."""
    assert residu_de_diffusion_cylindrique(sp.sin(TEMPS), 1, 1) != 0


def test_l_onde_amortie_de_l_effet_de_peau_satisfait_la_diffusion() -> None:
    """(15.14.2) a une dimension : `exp(-x / delta) cos(omega t - x / delta)`.

    Avec `delta = racine(2 rho / (mu omega))`, l'epaisseur de peau que la
    bibliotheque calculait deja.
    """
    mu, rho, omega = sp.symbols("mu rho_e omega", positive=True)
    peau = sp.sqrt(2 * rho / (mu * omega))
    courant = sp.exp(-ABSCISSE / peau) * sp.cos(omega * TEMPS - ABSCISSE / peau)
    assert residu_de_diffusion_plane(courant, mu, rho) == 0
    chiffree = peau.subs(
        {mu: PERMEABILITE_DU_VIDE, rho: 1 / CONDUCTIVITE_DU_CUIVRE, omega: 100 * sp.pi}
    )
    assert float(chiffree) == pytest.approx(epaisseur_de_peau(50.0))


def test_un_mode_de_bessel_s_eteint_dans_le_cylindre() -> None:
    """(15.14.3) : `J_0(k r) exp(-t / tau)` avec `k^2 = mu / (rho tau)`."""
    mu, rho, tau = sp.symbols("mu rho_e tau", positive=True)
    nombre = sp.sqrt(mu / (rho * tau))
    courant = sp.besselj(0, nombre * RAYON) * sp.exp(-TEMPS / tau)
    assert residu_de_diffusion_cylindrique(courant, mu, rho) == 0
