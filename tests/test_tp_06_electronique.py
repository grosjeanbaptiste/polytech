"""TP 6 d'electronique : les defauts de l'AOP et le dimensionnement.

Oracles : les formules du corrige officiel (`TP_6_Solution.pdf`) et les
valeurs typiques des fiches techniques qu'il reproduit (LM741, CA3140A,
AD8129, OPA549, OPA132), refaites en arithmetique ordinaire. La
bibliotheque, elle, resout les montages avec leurs sources de defaut
(tension de decalage, courants de polarisation) et un AOP de gain fini.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electronique.application import decalage_tp_06 as dc
from polytech.electronique.application import dimensionnement_tp_06 as dm
from polytech.electronique.application import polarisation_tp_06 as po
from polytech.electronique.domain.aop import aop
from polytech.electronique.domain.netliste import Admittance, SourceDeTension


def _sortie_decalee(e0: float, i_plus: float, i_moins: float) -> float:
    """`(1 + R2/R1) e0 - (1 + R2/R1) Rs I+ + R2 I-`, page 5 de l'officiel."""
    rs, r1, r2 = 10e3, 1e3, 9e3
    return (1 + r2 / r1) * e0 - (1 + r2 / r1) * rs * i_plus + r2 * i_moins


def test_ex4_un_etage_de_gain_quinze_garde_cent_kilohertz() -> None:
    assert math.isclose(dm.bande_d_un_etage(15), 1.5e6 / 15)


def test_ex4_la_regle_du_cours_donne_deux_etages() -> None:
    assert dm.etages_selon_le_cours() == 2


def test_ex4_la_chaine_entiere_en_demande_trois() -> None:
    assert dm.etages_selon_la_chaine() == 3


def test_ex5_les_courants_d_entree_du_lm741() -> None:
    """`I+ = Ibias + Ioffset/2`, `I- = Ibias - Ioffset/2`."""
    i_plus, i_moins = dc.courants_d_entree(80e-9, 20e-9)
    assert math.isclose(i_plus, 90e-9)
    assert math.isclose(i_moins, 70e-9)


def test_ex5_le_decalage_du_lm741() -> None:
    assert math.isclose(dc.decalage_lm741(), _sortie_decalee(1e-3, 90e-9, 70e-9))


def test_ex5_le_decalage_total_du_lm741_vaut_1_63_millivolt() -> None:
    """+10 mV, -9 mV et +0,63 mV : 1,63 mV au total."""
    e0, plus, moins = dc.decalage_lm741_par_source()
    assert (e0 * 1e3, plus * 1e3, moins * 1e3) == pytest.approx((10.0, -9.0, 0.63))
    assert dc.decalage_lm741() * 1e3 == pytest.approx(1.63, abs=5e-3)


def test_ex5_le_decalage_du_ca3140a() -> None:
    attendu = _sortie_decalee(2e-3, 10.25e-12, 9.75e-12)
    assert math.isclose(dc.decalage_ca3140a(), attendu)


def test_ex5_lu_en_picoamperes_le_lm741_rendrait_le_chiffre_de_l_officiel() -> None:
    """`9.99 mV` : la valeur qu'on obtient en prenant les nA pour des pA."""
    en_pico = dc.decalage_du_non_inverseur(1e-3, *dc.courants_d_entree(80e-12, 20e-12))
    assert round(en_pico * 1e3, 2) == 9.99


def test_ex6_le_point_de_repos_est_a_mi_alimentation() -> None:
    assert math.isclose(po.repos_de_sortie(), 2.5)


def test_ex6_le_gain_alternatif_est_celui_de_l_inverseur() -> None:
    assert math.isclose(po.gain_alternatif(), -100)


def test_ex6_le_gain_a_vingt_hertz_avec_quatre_vingts_microfarads() -> None:
    w = 2 * math.pi * 20
    h = -1j * w * 80e-6 * 100e3 / (1j * w * 1e3 * 80e-6 + 1)
    assert math.isclose(po.module_a_vingt_hertz(), abs(h))


def test_ex6_le_filtre_de_sortie_coupe_sous_vingt_hertz() -> None:
    assert math.isclose(
        po.coupure_du_filtre_de_sortie(), 1 / (2 * math.pi * 100e3 * 800e-9)
    )


def test_ex7_la_bande_de_l_inverseur_selon_le_gain_de_bruit() -> None:
    """Pole en boucle fermee : `GBW R1 / (R1 + R2)`, soit `GBW / 6`."""
    assert math.isclose(dm.bande_de_l_inverseur(10e3, 50e3, 1.5e6), 1.5e6 / 6)


def test_ex7_le_decalage_avec_un_millivolt() -> None:
    assert math.isclose(dc.decalage_de_l_exercice_7(), 6 * 1e-3 + 50e3 * 70e-9)


def test_ex7_le_chiffre_de_l_officiel_suppose_cinq_millivolts() -> None:
    assert dc.decalage_de_l_exercice_7(5e-3) * 1e3 == pytest.approx(33.5)
    assert dc.decalage_de_l_exercice_7() * 1e3 == pytest.approx(9.5)


def test_ex8_le_lm741_suffit() -> None:
    assert dm.aop_de_l_exercice_8() == "LM741"


def test_ex9_gains_maximaux_a_deux_megahertz() -> None:
    gains = dm.gains_maximaux_a_deux_megahertz()
    assert math.isclose(gains["AD8129"], 100)
    assert math.isclose(gains["OPA549"], 0.45)
    assert math.isclose(gains["OPA132"], 4)


def test_ex9_la_pente_requise() -> None:
    assert math.isclose(dm.pente_requise(), 2 * math.pi * 2e6 * 4 / 1e6)


def test_ex9_le_suiveur_opa549_a_deux_megahertz() -> None:
    h = 1 / (1 + 1j * 2e6 / 0.9e6)
    module, phase = dm.suiveur_opa549_a_deux_megahertz()
    assert math.isclose(module, abs(h))
    assert math.isclose(phase, math.degrees(cmath.phase(h)))


def test_ex4_la_chaine_de_deux_etages_perd_la_bande() -> None:
    """Deux etages de racine de 150 : `122.5 kHz sqrt(sqrt(2) - 1)`."""
    attendu = 1.5e6 / math.sqrt(150) * math.sqrt(math.sqrt(2) - 1)
    assert math.isclose(dm.bande_de_la_chaine(2), attendu)


def test_ex5_les_trois_parts_du_decalage_du_lm741() -> None:
    e0, plus, moins = dc.decalage_lm741_par_source()
    assert math.isclose(e0, 10 * 1e-3)
    assert math.isclose(plus, -10 * 10e3 * 90e-9)
    assert math.isclose(moins, 9e3 * 70e-9)


def test_une_contre_reaction_sur_l_entree_plus_est_refusee() -> None:
    """Meme module de pole, mais dans le demi-plan droit : instable."""
    circuit = [
        SourceDeTension(1, 0, dm.VIN, sp.Symbol("I")),
        Admittance(1, 2, sp.Rational(1, 10_000)),
        Admittance(2, 3, sp.Rational(1, 50_000)),
        *aop(2, 0, 3),
    ]
    with pytest.raises(ValueError, match="pole instable"):
        dm.pole_en_boucle_fermee(circuit, 3, 3, 1.5e6)
