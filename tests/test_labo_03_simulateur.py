"""Le laboratoire III d'electromagnetisme, recalcule sans simulateur.

Le labo fait verifier sous QUCS deux schemas de Thevenin, une reponse
indicielle et un filtre. Ce fichier refait chaque resultat par la
bibliotheque, et le recoupe :

- le montage simple, contre le corrige officiel : 4 V et 10/3 ohms ;
- le montage a dix resistances, que l'officiel laisse en blanc : la
  resistance par Kennelly, a la main, contre le resolveur ; la tension
  par superposition, contre le resolveur ; et Norton pour les lier ;
- le circuit R-L, dont la tension de bobine part de `v0` et s'eteint en
  `L/R` ;
- le filtre, passe-haut, a `|H| = 1/racine 2` en `f = R / (2 pi L)`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.application.enonces_labo_03 import (
    circuit_a_dix_resistances,
    circuit_simple,
    frequence_de_coupure,
    module_du_filtre,
    resistance_par_kennelly,
    tension_de_bobine,
)
from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    resistance_vue_des_bornes,
    tension_a_vide,
)
from polytech.electromagnetisme.domain.symboles import TEMPS


def test_le_montage_simple_donne_les_quatre_volts_de_l_officiel() -> None:
    assert tension_a_vide(circuit_simple(), "a", "m") == 4


def test_le_montage_simple_donne_les_dix_tiers_d_ohm_de_l_officiel() -> None:
    assert resistance_vue_des_bornes(circuit_simple(), "a", "m") == sp.Rational(10, 3)


def test_kennelly_a_la_main_rejoint_le_resolveur() -> None:
    reseau = circuit_a_dix_resistances()
    assert resistance_par_kennelly() == resistance_vue_des_bornes(reseau, "a", "b")


def test_la_resistance_de_thevenin_vaut_2_282_ohms() -> None:
    assert float(resistance_par_kennelly()) == pytest.approx(2.2819, abs=1e-4)


def test_la_superposition_rejoint_la_tension_a_vide() -> None:
    parts = [
        tension_a_vide(circuit_a_dix_resistances(*sources), "a", "b")
        for sources in ((10, 0, 0), (0, 5, 0), (0, 0, 1))
    ]
    total = tension_a_vide(circuit_a_dix_resistances(), "a", "b")
    assert sum(parts) == total
    assert float(total) == pytest.approx(1.4203, abs=1e-4)


def test_norton_lie_les_trois_mesures() -> None:
    reseau = circuit_a_dix_resistances()
    produit = resistance_vue_des_bornes(reseau, "a", "b") * courant_de_court_circuit(
        reseau, "a", "b"
    )
    assert sp.simplify(produit - tension_a_vide(reseau, "a", "b")) == 0


def test_la_tension_de_bobine_part_de_v0_et_s_eteint_en_l_sur_r() -> None:
    v0, r, inductance = sp.symbols("v_0 R L", positive=True)
    attendu = v0 * sp.exp(-r * TEMPS / inductance)
    assert sp.simplify(tension_de_bobine(v0, r, inductance) - attendu) == 0


def test_le_filtre_vaut_un_sur_racine_de_deux_a_la_coupure() -> None:
    coupure = frequence_de_coupure(2200, sp.Rational(484, 10000))
    assert float(coupure) == pytest.approx(7234.3, abs=0.1)
    module = module_du_filtre(2200, sp.Rational(484, 10000), coupure)
    assert float(module) == pytest.approx(2**-0.5)


def test_le_filtre_est_un_passe_haut() -> None:
    inductance = sp.Rational(484, 10000)
    assert module_du_filtre(2200, inductance, 0) == 0
    tres_haut = module_du_filtre(2200, inductance, 10**7)
    assert float(tres_haut) == pytest.approx(1.0, abs=1e-4)
