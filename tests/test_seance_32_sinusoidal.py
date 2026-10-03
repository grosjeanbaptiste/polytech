"""La seance 3.2 : regime sinusoidal et transitoires, recoupes par les noeuds.

Le manuscrit ecrit ses mailles en complexe puis les resout a la main,
arrondissant en chemin. Ce fichier resout les memes circuits par la loi
des noeuds -- en domaine de Laplace, restreint a `p = j omega` pour le
regime sinusoidal -- et compare aux **reponses imprimees du recueil**
(`Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`, section 3.3.3,
exercices 1, 2, 5, 6 et 7) et au manuscrit.

Trois desaccords y sont etablis, chacun par un chemin independant :

- exercice 1 : le manuscrit trouve 24,09 A, le calcul exact 24,12 A ;
  refaire son calcul avec ses propres intermediaires arrondis redonne
  ses chiffres, donc l'ecart vient des arrondis ;
- exercice 6 : le recueil imprime -163,71 degres ; le manuscrit et le
  calcul trouvent -163,38 et -163,375, et rien ne reproduit l'imprime ;
- exercice 7 : la derniere ligne du manuscrit porte un `+` devant
  l'exponentielle, qui viole `i(0) = 0` ; le recueil porte le `-`.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application.enonces_seance_32 import (
    BOBINE_DU_CINQ,
    PRIMAIRE_DU_SIX,
    SOURCE_DU_CINQ,
    branche_du_cinq,
    circuit_du_cinq,
    circuit_du_six,
    circuit_du_un,
    courant_secondaire_du_six,
)
from polytech.electromagnetisme.application.enonces_seance_32_transitoires import (
    courant_du_deux,
    courant_du_sept,
    tension_initiale_du_deux,
)
from polytech.electromagnetisme.application.macros_seance_32 import (
    puissances_des_sources,
)
from polytech.electromagnetisme.application.symboles_seance_32 import (
    C3,
    COURANT_TROIS_DU_CINQ,
    COURANT_UN_DU_CINQ,
    E0,
    E1,
    E3,
    L1,
    L2,
    OMEGA_DU_SIX,
    OMEGA_DU_UN,
    R0,
    R1,
    R2,
    R3,
    SIX_R2,
    THETA,
    THETA0,
    U0,
    UN_R,
    UN_RETARD,
    C,
    E,
    L,
    R,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import (
    en_regime_sinusoidal,
    puissance_complexe,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE


def _complexe(expression: sp.Expr) -> complex:
    """La valeur numerique d'un phaseur exact."""
    return complex(sp.N(expression, 30))


def _degres(valeur: complex) -> float:
    return math.degrees(cmath.phase(valeur))


def _courant_du_un() -> complex:
    return _complexe(
        en_regime_sinusoidal(circuit_du_un().potentiels()["h"], OMEGA_DU_UN) / UN_R
    )


def test_exercice_1_le_courant_exact_est_de_24_12_amperes() -> None:
    assert abs(_courant_du_un()) == pytest.approx(24.119, abs=1e-3)


def test_exercice_1_le_courant_retarde_de_10_21_degres_sur_e1() -> None:
    assert _degres(_courant_du_un()) == pytest.approx(-10.206, abs=1e-3)


def test_exercice_1_les_chiffres_du_manuscrit_viennent_de_ses_arrondis() -> None:
    """Son equation (I), refaite avec ses intermediaires : 23,71 - 4,26 j."""
    gauche = 220 - (219.29 - 9.57j)
    reactance = 1j * (3.14 + 2.51)
    courant = (gauche + reactance * (-3.81 - 87.37j)) / (
        reactance * (1 - 3.59j) - 2.51j
    )
    assert courant.real == pytest.approx(23.71, abs=5e-3)
    assert courant.imag == pytest.approx(-4.26, abs=5e-3)
    assert abs(courant) == pytest.approx(24.09, abs=5e-3)


def test_exercice_1_la_charge_absorbe_5236_watts() -> None:
    courant = _courant_du_un()
    assert float(UN_R) * abs(courant) ** 2 == pytest.approx(5235.7, abs=0.1)


def test_exercice_5_le_systeme_du_recueil_est_verifie_par_le_reseau() -> None:
    """Les deux equations imprimees, satisfaites par les courants du reseau."""
    reseau = circuit_du_cinq()
    i1 = en_regime_sinusoidal(reseau.courant(BOBINE_DU_CINQ), PULSATION)
    i3 = -en_regime_sinusoidal(reseau.courant(SOURCE_DU_CINQ), PULSATION)
    jw = sp.I * PULSATION
    premiere = (R1 + R2 + jw * L1) * i1 - R2 * i3 - E1
    seconde = R2 * i1 - (R2 + R3 / (1 + jw * R3 * C3)) * i3 - E3
    assert sp.simplify(premiere) == 0
    assert sp.simplify(seconde) == 0


def test_exercice_5_la_branche_absorbe_162_32_watts_et_28_83_vars() -> None:
    tension, courant = branche_du_cinq()
    puissance = _complexe(puissance_complexe(tension, courant))
    assert puissance.real == pytest.approx(162.322, abs=1e-3)
    assert puissance.imag == pytest.approx(28.828, abs=1e-3)


def test_exercice_5_le_dephasage_de_i1_sur_i3_vaut_191_31_degres() -> None:
    ecart = _degres(_complexe(COURANT_UN_DU_CINQ)) - _degres(
        _complexe(COURANT_TROIS_DU_CINQ)
    )
    assert ecart == pytest.approx(191.310, abs=1e-3)


def _courants_du_six() -> tuple[complex, complex]:
    reseau = circuit_du_six()
    primaire = en_regime_sinusoidal(reseau.courant(PRIMAIRE_DU_SIX), OMEGA_DU_SIX)
    secondaire = en_regime_sinusoidal(courant_secondaire_du_six(), OMEGA_DU_SIX)
    return _complexe(primaire), _complexe(secondaire)


def test_exercice_6_i1_avance_de_163_375_degres_sur_i2() -> None:
    """Le manuscrit ecrit -163,38 ; le recueil imprime -163,71."""
    primaire, secondaire = _courants_du_six()
    assert _degres(primaire / secondaire) == pytest.approx(-163.375, abs=1e-3)


def test_exercice_6_la_branche_r2_c2_absorbe_3_155_milliwatts() -> None:
    _, secondaire = _courants_du_six()
    assert float(SIX_R2) * abs(secondaire) ** 2 == pytest.approx(3.1549e-3, abs=1e-7)


def test_exercice_2_la_transformee_est_celle_du_recueil() -> None:
    """Facteur `C` compris -- la boite du manuscrit le perd."""
    attendu = (
        C
        * U0
        * (VARIABLE * L2 + R1 + R2)
        / (
            VARIABLE**2 * (L2 * R * C + L2 * R1 * C)
            + VARIABLE * (L2 + R * R1 * C + R * R2 * C + R1 * R2 * C)
            + R1
            + R2
        )
    )
    assert sp.simplify(courant_du_deux() - attendu) == 0


def test_exercice_2_la_tension_initiale_est_celle_du_recueil() -> None:
    attendu = (
        E0
        * sp.sqrt(2)
        / sp.sqrt(1 + (PULSATION * C * R0) ** 2)
        * sp.cos(THETA0 - sp.atan(PULSATION * C * R0))
    )
    ecart = tension_initiale_du_deux() - attendu
    assert sp.simplify(sp.expand_trig(ecart)) == 0


def test_exercice_7_le_courant_part_de_zero() -> None:
    assert sp.simplify(courant_du_sept().subs(TEMPS, 0)) == 0


def test_exercice_7_le_plus_du_manuscrit_viole_la_condition_initiale() -> None:
    """Avec `+`, `i(0) = 2 A cos(theta - phi)` : nul pour une seule phase."""
    phi = sp.atan(PULSATION * L / R)
    amplitude = E * sp.sqrt(2) / sp.sqrt(R**2 + (PULSATION * L) ** 2)
    manuscrit = amplitude * (
        sp.cos(PULSATION * TEMPS + THETA - phi)
        + sp.cos(THETA - phi) * sp.exp(-R * TEMPS / L)
    )
    assert (
        sp.simplify(manuscrit.subs(TEMPS, 0) - 2 * amplitude * sp.cos(THETA - phi)) == 0
    )


def test_exercice_1_les_sources_fournissent_ensemble_la_puissance_de_la_charge() -> (
    None
):
    """Les inductances n'absorbent aucun actif : `P_1 + P_2 = R I^2`."""
    premiere, seconde = puissances_des_sources(UN_RETARD)
    assert premiere + seconde == pytest.approx(5235.7, abs=0.1)


def test_exercice_1_le_retard_de_e2_renverse_le_partage() -> None:
    """Sans retard, `e_2`, moins inductif, fournirait le plus ; avec, `e_1`."""
    avec = puissances_des_sources(UN_RETARD)
    sans = puissances_des_sources(sp.Integer(0))
    assert avec[0] > avec[1]
    assert sans[0] < sans[1]
