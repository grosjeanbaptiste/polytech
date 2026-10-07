"""Signaux et systemes : le recueil d'exercices, circuits et reglages.

Source : `SignauxSystemes/EXS/Exs_Recueil.pdf`. Ce fichier couvre les
planches 7, 8, 10, 15, 16 et 19. Les reponses manuscrites des planches 15,
16 et 19 sont recopiees comme oracles ; celle de la planche 16 repose sur
une approximation, que le calcul exact corrige de 5 %.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.signaux.application import recueil_circuits as ci
from polytech.signaux.application import recueil_trottinette as tr
from polytech.signaux.domain.symboles import IMAGE as P

# --- Planches 7, 8 et 10 : celles de la seance « Chap. 4-5 » ----------------


def test_p7_les_trois_transmittances() -> None:
    """`1/(1 + RCp)`, `p/(p + R/L)` et `p/(p^2 + 20p + 200)`."""
    rc, rl, graphe = ci.transmittances_de_la_planche_7()
    assert sp.simplify(rc - 1 / (1 + P / 10**7)) == 0
    assert sp.simplify(rl - P / (P + 10**5)) == 0
    assert sp.simplify(graphe - P / (P**2 + 20 * P + 200)) == 0


def test_p8_la_condition_d_oscillation() -> None:
    """`R > (1/2) racine(L/C)`, puis `tau = 2RC`."""
    assert (
        sp.simplify(ci.resistance_limite_du_parallele() ** 2 - ci.L / (4 * ci.C)) == 0
    )
    constante, _ = ci.constante_et_pulsation_du_parallele()
    assert sp.simplify(constante - 2 * ci.R * ci.C) == 0


def test_p10_l_impulsionnelle_du_graphe() -> None:
    """`h(t) = e^(-10t) (cos 10t - sin 10t)` : part de 1, s'annule en `pi/40`."""
    assert ci.impulsionnelle_du_graphe(0) == 1
    assert ci.impulsionnelle_du_graphe(sp.pi / 40) == 0


# --- Planche 15 : une bobine shuntee, puis une resistance -------------------


def test_p15_la_transmittance_du_manuscrit() -> None:
    """Planche 15 : `(R^2 + pRL)/(R^2 + 2pRL)`, soit `(1/2)(p + R/L)/(p + R/2L)`."""
    attendue = (ci.R**2 + P * ci.R * ci.L) / (ci.R**2 + 2 * P * ci.R * ci.L)
    assert sp.simplify(ci.transmittance_de_la_planche_15() - attendue) == 0


def test_p15_les_limites_du_manuscrit() -> None:
    """Planche 15 : `a(0+) = 1/2`, `a(infini) = 1`."""
    assert ci.limites_de_l_indicielle_15() == (sp.Rational(1, 2), 1)


def test_p15_les_limites_par_le_raisonnement_physique() -> None:
    """Second chemin : bobine ouverte au depart, fil a la fin."""
    assert ci.limites_physiques_15() == ci.limites_de_l_indicielle_15()


def test_p15_la_constante_de_temps_du_manuscrit() -> None:
    """Planche 15 : `tau = 2L/R` ; `2 x 10^-4` s si `L/R = 10^-4` s."""
    assert sp.simplify(ci.constante_de_temps_15() - 2 * ci.L / ci.R) == 0
    assert float(
        ci.constante_de_temps_15().subs({ci.R: 100, ci.L: 0.01})
    ) == pytest.approx(2e-4)


def test_p15_l_indicielle() -> None:
    """`a(t) = 1 - (1/2) e^(-t/tau)` : elle part de 1/2 et monte vers 1."""
    instant = sp.Symbol("t", positive=True)
    attendue = 1 - sp.exp(-instant * ci.R / (2 * ci.L)) / 2
    assert sp.simplify(ci.indicielle_15(instant) - attendue) == 0


def test_p15_le_graphe_de_fluence() -> None:
    """Gain direct 1/2, puis un integrateur : gain `R/2L`, retour `-R/2L`."""
    direct, gain, retour = ci.gains_du_graphe_15()
    assert (direct, gain, retour) == (
        sp.Rational(1, 2),
        ci.R / (2 * ci.L),
        -ci.R / (2 * ci.L),
    )
    par_mason = (direct + gain / P) / (1 - retour / P)
    assert sp.simplify(par_mason - ci.transmittance_de_la_planche_15()) == 0


# --- Planche 16 : la trottinette et la bordure ------------------------------


def test_p16_la_transmittance_du_manuscrit() -> None:
    """Planche 16 : `H = beta/(p^2 + gamma p + 1)`."""
    attendue = tr.BETA / (P**2 + tr.GAMMA * P + 1)
    assert sp.simplify(tr.transmittance() - attendue) == 0


def test_p16_le_plancher_ne_bouge_pas_d_abord() -> None:
    """`a(0+) = 0`, et meme `a'(0+) = 0` : ni saut, ni pente."""
    assert tr.depart_de_l_indicielle() == (0, 0)


def test_p16_beta_vaut_un() -> None:
    """Planche 16 : `a(infini) = beta`, donc `beta = 1`."""
    assert tr.beta() == 1


def test_p16_le_gamma_du_manuscrit() -> None:
    """Planche 16 : `gamma = racine(4/5)`, en prenant `3 pi / 10` pour un."""
    assert tr.gamma_approche() == sp.sqrt(sp.Rational(4, 5))


def test_p16_le_gamma_exact() -> None:
    """Sans l'approximation : `10 / racine(9 pi^2 + 25) = 0,937`."""
    assert sp.simplify(tr.gamma_exact() - 10 / sp.sqrt(9 * sp.pi**2 + 25)) == 0
    assert float(tr.gamma_exact()) == pytest.approx(0.937, abs=5e-4)


def test_p16_l_ecart_entre_les_deux() -> None:
    """Le manuscrit sous-estime `gamma` de 4,6 % : 1,59 oscillation au lieu de 1,5."""
    assert tr.oscillations_avant_le_regime(tr.gamma_exact()) == pytest.approx(1.5)
    assert tr.oscillations_avant_le_regime(tr.gamma_approche()) == pytest.approx(
        1.59, abs=5e-3
    )
    ecart = 1 - float(tr.gamma_approche() / tr.gamma_exact())
    assert ecart == pytest.approx(0.046, abs=5e-4)


# --- Planche 19 : une bobine, puis R et C en parallele ----------------------


def test_p19_la_transmittance_du_manuscrit() -> None:
    """Planche 19 : `(1/LC)/(p^2 + p/RC + 1/LC)`."""
    produit = ci.L * ci.C
    attendue = (1 / produit) / (P**2 + P / (ci.R * ci.C) + 1 / produit)
    assert sp.simplify(ci.transmittance_de_la_planche_19() - attendue) == 0


def test_p19_les_limites_du_manuscrit() -> None:
    """Planche 19 : `h` va de 0 a 0, `a` de 0 a 1."""
    assert ci.limites_19() == {"h": (0, 0), "a": (0, 1)}


def test_p19_la_resistance_limite() -> None:
    """Planche 19 : `(1/2) racine(10^-3/10^-7) = 50` ohms."""
    assert ci.resistance_limite_19() == 50


def test_p19_le_graphe_du_manuscrit() -> None:
    """Planche 19 : retours `-2 x 10^5` et `-10^10`, gains directs 0 et `10^10`."""
    assert ci.gains_du_graphe_19() == ((-2 * 10**5, -(10**10)), (0, 10**10))


def test_p19_a_cinquante_ohms_le_pole_est_double() -> None:
    """La plus petite valeur est le regime critique : elle n'oscille pas encore."""
    assert ci.poles_19(50) == {-(10**5): 2}
    assert math.isclose(
        float(sp.im(next(iter(ci.poles_19(51))))) ** 2, 1e10 - (1e7 / 102) ** 2
    )
