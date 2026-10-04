"""TP 5 d'electronique : l'amplification, du diviseur au montage a trois AOP.

Les oracles sont les formules du zero virtuel que donne le corrige
officiel (`TP_5_Solution.pdf`), refaites ici en arithmetique ordinaire.
La bibliotheque, elle, resout chaque montage avec des AOP a gain fini
puis fait tendre le gain vers l'infini : les deux chemins n'ont en commun
que le schema.
"""

from __future__ import annotations

import math

from polytech.electronique.application import amplification_tp_05 as am
from polytech.electronique.application import combinaisons_tp_05 as cb
from polytech.electronique.application import montages_tp_05 as mt
from polytech.electronique.application import saturation_tp_05 as sa


def test_ex1_le_diviseur_rend_quatre_centiemes() -> None:
    assert math.isclose(am.gain_du_diviseur(), 3 / 75)


def test_ex2_l_amplificateur_reel_perd_du_gain() -> None:
    vi = 0.5 * 10e3 / (200 + 10e3)
    vo = 20 * vi * 1e3 / (75 + 1e3)
    entree, sortie = am.amplificateur_reel()
    assert math.isclose(entree, vi)
    assert math.isclose(sortie, vo)


def test_ex7_l_amplificateur_differentiel() -> None:
    assert math.isclose(am.sortie_differentielle(), 100 * (18.3 - 18.2))


def test_ex8_la_boucle_ouverte_sature_presque_toujours() -> None:
    cas = sa.boucle_ouverte()
    assert math.isclose(cas["a"], 0.1)
    assert cas["b"] == -12
    assert cas["d"] == 12
    assert cas["e"] == 12
    assert math.isclose(cas["f"], 0.1)


def test_ex8c_la_premiere_demi_periode_est_au_rail_negatif() -> None:
    """`V- = sin > 0` sur la premiere demi-periode : `A (0 - V-) < 0`."""
    assert sa.boucle_ouverte()["c"] == -12


def test_ex10_les_deux_gains() -> None:
    inverseur, non_inverseur = mt.gains_de_l_exercice_10()
    assert inverseur == -10
    assert non_inverseur == 11


def test_ex10b_moins_quatre_volts_saturent_les_deux() -> None:
    assert sa.exercice_10b() == (12, -12)


def test_ex11_le_non_inverseur() -> None:
    assert mt.gain_de_l_exercice_11() == 6


def test_ex12_r3_ne_change_pas_le_gain_lineaire() -> None:
    for r3 in (1e3, 20e3):
        assert math.isclose(mt.gain_de_l_exercice_12(r3), -10)


def test_ex12_le_seuil_de_saturation() -> None:
    """La sortie de l'AOP vaut `-(R2 + R3)/R1 Vin` : elle atteint 12 V."""
    assert math.isclose(sa.seuil_de_l_exercice_12(1e3), 12 / 11)
    assert math.isclose(sa.seuil_de_l_exercice_12(20e3), 12 / 30)


def test_ex12_la_sortie_saturee_suit_la_formule_officielle() -> None:
    """`Vo = 12 + (Vin - 12) R3 / (R1 + R2 + R3)`, a `Vin = -2 V`."""
    for r3 in (1e3, 20e3):
        attendu = 12 + (-2 - 12) * r3 / (1e3 + 10e3 + r3)
        assert math.isclose(sa.crete_de_l_exercice_12(r3), attendu)


def test_ex13_gain_et_courant_de_charge() -> None:
    gain, courant = mt.exercice_13()
    assert math.isclose(gain, 1 + 1 / 5)
    assert math.isclose(courant, 0.5 * 1.2 / 1e3)


def test_ex14_le_suiveur_adapte_l_impedance() -> None:
    direct, tamponne = mt.exercice_14()
    assert math.isclose(direct, 0.5)
    assert math.isclose(tamponne, 1)


def test_ex15_les_deux_inverseurs() -> None:
    simple, thevenin = mt.exercice_15()
    assert math.isclose(simple, -5 / (1 + 1))
    assert math.isclose(thevenin, -5 / (0.5 + 0.5 + 1) * 0.5)


def test_ex16_les_deux_combinaisons() -> None:
    (a1, a2), (b1, b2) = cb.exercice_16()
    assert math.isclose(a1, (1 + 10 / 5) * 10 / 11)
    assert math.isclose(a2, -2)
    assert math.isclose(b1, -25 / 10)
    assert math.isclose(b2, -25 / 5)


def test_ex17_les_coefficients_des_deux_entrees() -> None:
    assert cb.coefficients_de_l_exercice_17(5e3) == (6, -5)
    assert cb.coefficients_de_l_exercice_17(20e3) == (21, -20)


def test_ex18_thevenin_puis_inverseur() -> None:
    assert math.isclose(cb.exercice_18(), -5 / (1 + 4) * 0.5e-3 * 4e3)


def test_ex18_la_source_court_circuitee_ne_fait_rien() -> None:
    assert cb.exercice_18_court_circuit() == 0


def test_ex19_composantes_continue_et_alternative() -> None:
    continu, alternatif = cb.exercice_19()
    assert math.isclose(continu, 5 * 1e-6 * 1e3)
    assert math.isclose(alternatif, -40)


def test_ex12_la_sortie_de_l_aop_porte_la_chute_dans_r3() -> None:
    """`V_s = -(R2 + R3) / R1 V_in`."""
    assert math.isclose(sa.sortie_de_l_aop_12(1e3), -(10 + 1) / 1)
    assert math.isclose(sa.sortie_de_l_aop_12(20e3), -(10 + 20) / 1)


def test_ex15_le_thevenin_de_l_entree() -> None:
    rapport, resistance = mt.thevenin_de_l_exercice_15()
    assert math.isclose(rapport, 1e3 / (1e3 + 1e3))
    assert math.isclose(resistance, 1e3 * 1e3 / (1e3 + 1e3))


def test_ex19_le_decalage_continu_de_l_entree_plus() -> None:
    assert math.isclose(cb.decalage_de_l_exercice_19(), 1e-6 * 1e3)
