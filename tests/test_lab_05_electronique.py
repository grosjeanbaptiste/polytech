"""Laboratoire 5 d'electronique : concevoir un amplificateur audio.

Oracles : l'enonce et le corrige officiel
(`ElectroniqueInstrumentation/LABOS/Lab_05_AOP_Conception_et_realisation_
Enonce.pdf` et `..._Solution.pdf`), dont chaque nombre est cite avec sa
question. Les schemas sont des images : ils ont ete lus sur les pages
rendues, et chaque etage est remonte par ses noeuds.

Les ecarts que le recalcul etablit :

* question 6 : 5,5 mV et -22,2 mV sont calcules avec `R_1 = 10 kohm` et
  `R_2 = 100 kohm`, un gain de 11 ; le premier etage du schema porte 1 kohm
  et 39 kohm, un gain de 40, d'ou 20 mV et -88 mV ;
* section 2.2.5 : « C = 995 uF » pour couper a 20 Hz sur 16 ohm ; la
  formule ecrite donne 497 uF -- 995 est la valeur pour 8 ohm ;
* section 2.2.6 : 573 Hz est la frequence ou le module vaut `1/racine(2)`
  dans l'absolu ; a -3 dB sous le plateau la coupure est a 605 Hz, la
  valeur que les questions 16 et 17 emploient ;
* l'etage de sortie en pont double la tension (question 12) : le gain
  maximal de l'amplificateur est 364, non les 160 du cahier des charges.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.application import chaine_lab_05 as ch
from polytech.electronique.application import decalage_lab_05 as dc
from polytech.electronique.application import dimensionnement_lab_05 as dm
from polytech.electronique.application import egaliseur_lab_05 as eg
from polytech.electronique.application import sortie_lab_05 as so

P = sp.Symbol("p")


# --- Questions 1 a 5 : le dimensionnement -----------------------------


def test_q1_la_crete_que_le_haut_parleur_supporte() -> None:
    """`P = V_eff^2 / R` : 2,83 V efficaces, soit 4 V de crete."""
    assert dm.tension_efficace_maximale() == pytest.approx(2 * math.sqrt(2))
    assert dm.crete_maximale() == pytest.approx(4.0)


def test_q2_a_q4_le_gain_et_le_nombre_d_etages() -> None:
    """160 a atteindre, 150 par etage au plus : deux etages."""
    assert dm.gain_maximal() == pytest.approx(160.0)
    assert dm.gain_maximal_par_etage() == pytest.approx(150.0)
    assert dm.etages() == 2


def test_les_gains_realises() -> None:
    """Section 2.2.3 : 40, puis 4,54 avec 22 kohm ; 181 au lieu de 160."""
    premier, second = dm.gains_realises()
    assert premier == pytest.approx(40.0)
    assert second == pytest.approx(100 / 22)
    assert premier * second == pytest.approx(181.8, abs=0.05)
    assert dm.bandes_des_etages() == pytest.approx((75e3, 3e6 / (1 + 100 / 22)))
    assert all(bande >= 20e3 for bande in dm.bandes_des_etages())


def test_le_courant_que_la_charge_demande() -> None:
    """Section 2.2.5 : 4 V sur 16 ohm, 250 mA."""
    assert dm.courant_maximal() == pytest.approx(0.25)


# --- Questions 6 et 7 : les imperfections -----------------------------


def test_q6_le_decalage_avec_les_resistances_du_schema() -> None:
    """Gain de 40 : 20 mV apres le premier etage, -88 mV apres le second."""
    premier, second = dc.decalages()
    assert premier * 1e3 == pytest.approx(20.0, abs=1e-3)
    assert second * 1e3 == pytest.approx(-88.14, abs=5e-3)
    attendu = -premier * 100 / 22 + 0.5e-3 * (1 + 100 / 22) + 100e3 * 0.5e-12
    assert second == pytest.approx(attendu, rel=1e-9, abs=0)


def test_q6_les_nombres_de_l_officiel_supposent_un_gain_de_onze() -> None:
    """5,5 mV et -22,2 mV : ceux de `R_1 = 10 kohm`, `R_2 = 100 kohm`."""
    premier, second = dc.decalages(vers_la_masse=10e3, contre_reaction=100e3)
    assert premier * 1e3 == pytest.approx(5.5, abs=5e-3)
    assert second * 1e3 == pytest.approx(-22.2, abs=0.05)


def test_q7_les_courants_d_entree_ne_pesent_rien() -> None:
    """Quelques dizaines de nanovolts, contre des dizaines de millivolts."""
    avec, sans = dc.decalages(), dc.decalages(courants=False)
    assert abs(avec[0] - sans[0]) < 1e-7
    assert abs(avec[1] - sans[1]) < 1e-6
    assert abs(avec[1] - sans[1]) > 0
    assert sans[0] == pytest.approx(40 * 0.5e-3, rel=1e-9, abs=0)
    du_premier = -600 * 1.5e-12 * 40 + 39e3 * 0.5e-12
    assert avec[0] - sans[0] == pytest.approx(du_premier, rel=1e-6, abs=0)
    du_second = -du_premier * 100 / 22 + 100e3 * 0.5e-12
    assert avec[1] - sans[1] == pytest.approx(du_second, rel=1e-6, abs=0)


# --- Questions 8 a 12 : l'etage de sortie -----------------------------


def test_q8_a_q10_l_etage_a_alimentation_unique() -> None:
    """`E` continue passe telle quelle, `V_i` continu non, `V_i` alternatif est inverse.

    Ce que la question ne demande pas : au-dela de la coupure, `E` est
    amplifiee par deux. D'ou le condensateur qui la stabilise.
    """
    de_e, de_vi = so.transferts(P)
    assert de_e.subs(P, 0) == 1
    assert sp.limit(de_e, P, sp.oo) == 2
    assert de_vi.subs(P, 0) == 0
    assert sp.limit(de_vi, P, sp.oo) == -1
    r, c = sp.nsimplify(so.R4), sp.nsimplify(so.C2)
    assert sp.simplify(de_vi + P * r * c / (1 + P * r * c)) == 0


def test_q11_la_coupure_du_passe_haut() -> None:
    """Corrige : 137,74 rad/s, soit 21,9 Hz."""
    assert so.pulsation_de_coupure() == pytest.approx(137.74, abs=5e-3)
    assert so.frequence_de_coupure() == pytest.approx(21.9, abs=0.05)
    de_vi = so.transferts(P)[1]
    module = abs(complex(de_vi.subs(P, sp.I * sp.nsimplify(so.pulsation_de_coupure()))))
    assert module == pytest.approx(1 / math.sqrt(2))


def test_le_condensateur_de_liaison_qu_il_faudrait() -> None:
    """Enonce : « 995 uF » ; `1 / (2 pi R f)` donne 497 uF sur 16 ohm a 20 Hz."""
    assert so.condensateur_de_liaison() * 1e6 == pytest.approx(497.4, abs=0.05)
    assert so.condensateur_de_liaison(resistance=8.0) * 1e6 == pytest.approx(
        995.0, abs=0.5
    )


def test_q12_le_montage_en_pont() -> None:
    """`V_o1 = E - V_i`, `V_o2 = E + V_i`, et `V_o2 - V_o1 = 2 V_i`."""
    e, vi = sp.symbols("E V_i")
    premiere, seconde = so.sorties_du_pont(e, vi)
    assert sp.simplify(premiere - (e - vi)) == 0
    assert sp.simplify(seconde - (e + vi)) == 0
    assert sp.simplify(seconde - premiere - 2 * vi) == 0


def test_l_excursion_et_la_puissance_du_pont() -> None:
    """Chaque sortie reste entre 0 et 5 V : 2,5 V de crete, 5 V sur la charge."""
    assert so.crete_admissible_a_l_entree() == pytest.approx(2.5)
    assert so.crete_maximale_sur_la_charge() == pytest.approx(5.0)
    assert so.puissance_maximale_sur_la_charge() == pytest.approx(25 / 32)
    assert so.puissance_maximale_sur_la_charge() > 0.5


# --- Question 13 et section 2.2.6 : l'egaliseur -----------------------


def test_la_frequence_de_separation_est_une_moyenne_geometrique() -> None:
    """Du Do0 (16,35 Hz) au Do10 (16 742 Hz) : le Do5, 523 Hz."""
    assert eg.frequence_de_separation() == pytest.approx(523.2, abs=0.05)
    assert eg.frequence_de_separation() == pytest.approx(440 * 2 ** (3 / 12), rel=2e-4)


def test_le_filtre_des_basses() -> None:
    """Section 2.2.6 : `-R_5 / (R_3 + R_4 + p C_1 R_3 R_4)`."""
    r3, r4, r5, c1 = (sp.nsimplify(v) for v in (eg.R3, eg.R4, eg.R5, eg.C1))
    attendu = -r5 / (r3 + r4 + P * c1 * r3 * r4)
    assert sp.simplify(eg.basses(P) - attendu) == 0
    assert eg.plateau_des_basses() == pytest.approx(100 / 102.7)


def test_les_deux_coupures_du_filtre_des_basses() -> None:
    """573 Hz ou le module vaut `1/racine(2)` ; 605 Hz a -3 dB sous le plateau."""
    absolue, relative = eg.coupures_des_basses()
    assert absolue == pytest.approx(573.1, abs=0.05)
    assert relative == pytest.approx(605.4, abs=0.05)
    module = abs(complex(eg.basses(P).subs(P, 2j * math.pi * absolue)))
    assert module == pytest.approx(1 / math.sqrt(2))
    a_la_relative = abs(complex(eg.basses(P).subs(P, 2j * math.pi * relative)))
    assert a_la_relative == pytest.approx(eg.plateau_des_basses() / math.sqrt(2))
    assert relative == pytest.approx(1 / (2 * math.pi * 100e-9 * 2700 * 1e5 / 102700))


def test_le_filtre_des_aigus_a_la_meme_coupure() -> None:
    """`C_2` en serie, `R_6 // R_7` : 605 Hz encore, et un plateau de 1."""
    assert eg.coupure_des_aigus() == pytest.approx(eg.coupures_des_basses()[1])
    loin = abs(complex(eg.aigus(P).subs(P, 2j * math.pi * 1e6)))
    assert loin == pytest.approx(1.0, abs=1e-3)
    assert eg.aigus(P).subs(P, 0) == 0
    a_la_coupure = abs(
        complex(eg.aigus(P).subs(P, 2j * math.pi * eg.coupure_des_aigus()))
    )
    assert a_la_coupure == pytest.approx(1 / math.sqrt(2))


def test_q16_q17_les_gains_a_verifier() -> None:
    """Potentiometres au maximum : 0,96 a 100 Hz pour les basses, 0,86 a 1 kHz."""
    coupure = eg.coupure_des_aigus()
    basses = eg.plateau_des_basses() / math.hypot(1.0, 100 / coupure)
    assert eg.gain_des_basses(100.0) == pytest.approx(basses)
    assert basses == pytest.approx(0.9607, abs=5e-5)
    aigus = (1000 / coupure) / math.hypot(1.0, 1000 / coupure)
    assert eg.gain_des_aigus(1000.0) == pytest.approx(aigus)
    assert aigus == pytest.approx(0.8555, abs=5e-5)
    assert eg.gain_des_basses(6054.0) < 0.1
    assert eg.gain_des_aigus(60.54) < 0.1
    assert eg.gain_des_basses(100.0, potentiometre=0.0) == 0.0
    assert eg.gain_des_basses(100.0, potentiometre=50e3) == pytest.approx(basses / 2)


def test_basses_et_aigus_au_maximum_la_somme_est_plate() -> None:
    """`(R_4/(R_3+R_4) + j x) / (1 + j x)` : entre 0,974 et 1 a toute frequence."""
    for frequence in (20.0, 100.0, 605.4, 5e3, 20e3):
        assert 0.9737 <= eg.somme_au_maximum(frequence) <= 1.0
    assert eg.somme_au_maximum(1e-3) == pytest.approx(eg.plateau_des_basses())
    assert eg.ondulation_de_la_somme() == pytest.approx(0.231, abs=5e-4)


# --- Questions 15 a 20 : la chaine complete ---------------------------


def test_q18_le_sommateur() -> None:
    """Gain de 0 a -4,54 par voie ; l'enonce annonce « entre -4 et 0 »."""
    assert ch.gain_du_sommateur() == pytest.approx(-100 / 22)
    assert ch.gain_du_sommateur(potentiometre=0.0) == 0.0
    assert ch.bande_du_sommateur() == pytest.approx(3e6 / (1 + 100 / 11))


def test_q20_le_gain_maximal_de_l_amplificateur() -> None:
    """40, l'egaliseur, 4,54, puis le pont qui double : 364 et non 160."""
    assert ch.gain_maximal() == pytest.approx(40 * 100 / 22 * 2)
    assert ch.gain_maximal() == pytest.approx(363.6, abs=0.05)
    assert ch.gain_maximal() > 2 * 160
    assert ch.crete_demandee() == pytest.approx(25e-3 * ch.gain_maximal())
    assert ch.crete_demandee() > so.crete_maximale_sur_la_charge()


def test_q20_le_reglage_qui_respecte_le_haut_parleur() -> None:
    """Pour 4 V de crete depuis 25 mV : un gain de 160, soit 44 kohm au volume."""
    reglage = ch.potentiometre_pour_le_gain(160.0)
    assert reglage == pytest.approx(44e3)
    assert ch.gain_maximal(potentiometre=reglage) == pytest.approx(160.0)
    with pytest.raises(ValueError, match="gain hors de portee"):
        ch.potentiometre_pour_le_gain(400.0)


def test_q20_la_bande_passante() -> None:
    """-3,7 dB a 20 Hz (passe-haut de sortie a 21,9 Hz) ; -0,3 dB a 20 kHz."""
    passe_haut = 10 * math.log10(1 + (21.922 / 20) ** 2)
    plateau = -20 * math.log10(100 / 102.7)
    assert ch.perte(20.0) == pytest.approx(passe_haut + plateau, abs=0.01)
    assert ch.perte(20.0) == pytest.approx(3.66, abs=0.01)
    premier = 10 * math.log10(1 + (20 / 75) ** 2)
    assert ch.perte(20e3) == pytest.approx(premier, abs=0.03)
    assert ch.perte(20e3) == pytest.approx(0.32, abs=0.01)
    assert ch.perte(1e3) < 0.1
    bas, haut = ch.bande_a_trois_decibels()
    assert ch.perte(bas) == pytest.approx(3.0103, abs=1e-3)
    assert ch.perte(haut) == pytest.approx(3.0103, abs=1e-3)
    assert bas == pytest.approx(23.2, abs=0.1)
    assert 60e3 < haut < 75e3


def test_les_courbes_de_l_egaliseur() -> None:
    """Basses, aigus et leur somme, en decibels, de 20 Hz a 20 kHz."""
    frequences, basses, aigus, somme = eg.courbes()
    assert (frequences[0], frequences[-1]) == pytest.approx((20.0, 20e3))
    assert basses[0] == pytest.approx(20 * math.log10(eg.gain_des_basses(20.0)))
    assert aigus[-1] == pytest.approx(20 * math.log10(eg.gain_des_aigus(20e3)))
    assert max(somme) <= 0.0
    assert min(somme) >= -eg.ondulation_de_la_somme() - 1e-9
    assert len(frequences) == len(basses) == len(aigus) == len(somme) > 100


def test_q20_une_capacite_de_sortie_plus_grande_de_moitie_tient_vingt_hertz() -> None:
    """La coupure du passe-haut va comme `1 / C_3` : divisee par 1,5, elle suffit."""
    abaissee = so.frequence_de_coupure() / 1.5
    perte = 10 * math.log10(1 + (abaissee / 20) ** 2) + eg.ondulation_de_la_somme()
    assert perte < 10 * math.log10(2)
    assert ch.perte(20.0) > 10 * math.log10(2)
