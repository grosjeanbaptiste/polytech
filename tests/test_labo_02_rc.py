"""Le laboratoire II d'electromagnetisme : un R-C, de la theorie a la mesure.

Ce que l'oscilloscope doit montrer se calcule, et chaque calcul est
recoupe ici par un chemin independant :

- la charge `v0 (1 - e^{-t/RC})`, et l'impulsion par superposition de
  deux echelons, contre la transformee de Laplace inversee ;
- la periode et la duree du signal carre, prescrites a `20 tau` et `T/2` ;
- en sinusoidal, les deux phaseurs dont la somme doit redonner la source.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application.enonces_labo_02 import (
    CAPACITE,
    FREQUENCE,
    LECTURES_OFFICIELLES,
    RESISTANCE_DE_CHARGE,
    RESISTANCE_SINUS,
    charge_du_condensateur,
    constante_de_temps,
    phaseurs_du_sinus,
    reponse_a_l_impulsion,
)
from polytech.electromagnetisme.domain.circuits.premier_ordre import (
    constante_par_deux_points,
)
from polytech.electromagnetisme.domain.symboles import TEMPS


def test_la_charge_suit_la_forme_du_cours() -> None:
    v0, r, c = sp.symbols("v_0 R C", positive=True)
    attendu = v0 * (1 - sp.exp(-TEMPS / (r * c)))
    assert sp.simplify(charge_du_condensateur(v0, r, c) - attendu) == 0


def test_la_constante_de_temps_du_montage_vaut_cent_microsecondes() -> None:
    assert constante_de_temps() == pytest.approx(1e-4)
    produit = RESISTANCE_DE_CHARGE * CAPACITE
    assert produit == pytest.approx(1e-4)


def test_l_impulsion_est_la_difference_de_deux_echelons() -> None:
    """Contre la superposition ecrite a la main, en `t = 7 tau`."""
    tau = sp.Symbol("tau", positive=True)
    duree = 5 * tau
    reponse = reponse_a_l_impulsion(tau, duree)
    a_la_main = (1 - sp.exp(-7)) - (1 - sp.exp(-2))
    assert sp.simplify(reponse.subs(TEMPS, 7 * tau) - a_la_main) == 0


def test_l_impulsion_atteint_99_3_pour_cent_a_sa_fin() -> None:
    tau = sp.Symbol("tau", positive=True)
    valeur = reponse_a_l_impulsion(tau, 5 * tau).subs(TEMPS, 5 * tau)
    assert float(valeur) == pytest.approx(1 - math.exp(-5))


def test_les_deux_phaseurs_redonnent_la_source() -> None:
    """La loi des mailles en complexe : `V_R + V_C = V_s`."""
    resistance, condensateur = phaseurs_du_sinus()
    assert resistance + condensateur == pytest.approx(1.0)


def test_le_condensateur_retarde_de_15_46_degres() -> None:
    _, condensateur = phaseurs_du_sinus()
    assert abs(condensateur) == pytest.approx(0.9639, abs=1e-4)
    assert math.degrees(cmath.phase(condensateur)) == pytest.approx(-15.46, abs=0.01)


def test_la_resistance_avance_de_74_54_degres() -> None:
    resistance, _ = phaseurs_du_sinus()
    assert abs(resistance) == pytest.approx(0.2665, abs=1e-4)
    assert math.degrees(cmath.phase(resistance)) == pytest.approx(74.54, abs=0.01)


def test_les_donnees_du_sinus_sont_celles_de_l_enonce() -> None:
    assert (RESISTANCE_SINUS, FREQUENCE) == (2200, 2000)


def test_les_lectures_officielles_donnent_tau_a_quelques_pour_cent() -> None:
    """Deux points de decharge : 106,5 us, que l'officiel arrondit a 0,11 ms."""
    lu = LECTURES_OFFICIELLES
    tau = constante_par_deux_points(
        lu["premier_instant"],
        lu["premiere_tension"],
        lu["second_instant"],
        lu["seconde_tension"],
    )
    assert tau == pytest.approx(106.5e-6, abs=0.1e-6)


def test_le_temps_de_montee_officiel_donne_un_autre_tau() -> None:
    """208 us / ln 9 = 94,7 us : les deux mesures encadrent la theorie."""
    tau = LECTURES_OFFICIELLES["montee"] / math.log(9)
    assert tau == pytest.approx(94.7e-6, abs=0.1e-6)
