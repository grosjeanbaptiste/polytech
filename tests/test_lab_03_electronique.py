"""Laboratoire 3 d'electronique : dimensionner quatre chaines d'acquisition.

Oracles : le corrige officiel
(`ElectroniqueInstrumentation/LABOS/Lab_03_Instrumentation_Solution.pdf`,
quatorze pages), dont chaque nombre est cite avec sa page.

Les ecarts que le recalcul etablit :

* page 3 : « 100^2 » pour la densite de l'ampli B, qui vaut 1 nV -- le
  resultat imprime, `2 10^-8`, est bien celui de 1 nV ;
* page 4 : la somme des cinq puissances vaut `2,5402 10^-6` et non
  `2,504 10^-6` ; d'ou 1,59 mV et non 1,58, 11,27 uV et non 11,17 ;
* page 4 : « 8 x 40 kS/s = 360 kS/s », qui font 320 ;
* page 5 : « un seul ADC1 », pour l'ADC3 que la phrase decrit.
"""

from __future__ import annotations

import math

import pytest

from polytech.electronique.application import budget_1_2_lab_03 as budget
from polytech.electronique.application import chaine_1_1_lab_03 as un
from polytech.electronique.application import chaine_1_2_lab_03 as deux
from polytech.electronique.application import chaine_1_3_lab_03 as trois
from polytech.electronique.application import chaine_1_4_lab_03 as quatre
from polytech.electronique.application.donnees_lab_03 import (
    UnDeux,
    UnQuatre,
    UnTrois,
    UnUn,
)
from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    densite_de_bruit_maximale,
)
from polytech.electronique.domain.chaine_d_acquisition import bruit_efficace

# --- Exercice 1.1 -----------------------------------------------------


def test_ex11_les_gains() -> None:
    """Page 1 : 5000 en tout, 5 avant le passe-haut, 1000 apres."""
    total, premier, second = un.gains()
    assert (total, premier, second) == pytest.approx((5000.0, 5.0, 1000.0))
    assert un.gain_exact_du_premier_etage() == pytest.approx(5 / 1.001)
    assert un.sortie_maximale_du_premier_etage() == pytest.approx(5.005)


def test_ex11_les_amplificateurs() -> None:
    """Page 2 : gains maximaux de 50, 500 et 5000 sur 10 kHz."""
    assert un.gains_maximaux() == pytest.approx(
        {"AMP1": 50.0, "AMP2": 500.0, "AMP3": 5000.0}
    )
    assert un.amplificateurs_possibles(5.0) == ("AMP1", "AMP2", "AMP3")
    assert un.amplificateurs_possibles(1000.0) == ("AMP3",)
    assert un.amplificateurs_possibles(5000.0) == ("AMP3",)
    assert un.amplificateurs_possibles(5001.0) == ()


def test_ex11_les_convertisseurs() -> None:
    """Page 2 : dix bits au moins, 40 kHz au moins -- les CAN 5 et 6."""
    assert un.bits_minimaux() == 10
    assert math.log2(1000) == pytest.approx(9.97, abs=5e-3)
    assert un.cadence_minimale() == pytest.approx(40e3)
    assert un.convertisseurs_possibles() == ("CAN5", "CAN6")
    assert un.pas_a_l_entree(10) == pytest.approx(2e-3 / 1023)
    assert un.pas_a_l_entree(10) < UnUn.RESOLUTION < un.pas_a_l_entree(9)


def test_ex11_les_cinq_puissances_de_bruit() -> None:
    """Page 3 : `5 10^-7`, `2 10^-6`, `2 10^-8`, `2 10^-10`, `2 10^-8` volts carres."""
    puissances = un.puissances()
    assert list(puissances.values()) == pytest.approx(
        [5e-7, 2e-6, 2e-8, 2e-10, 2e-8], rel=1e-9, abs=0
    )
    assert list(puissances) == ["ampli A", "passe-haut", "ampli B", "passe-bas", "CAN"]


def test_ex11_la_somme_que_l_officiel_ecrit_n_est_pas_la_somme() -> None:
    """Page 4 : « 2,504 » ; les cinq termes font 2,5402."""
    total, efficace, densite = un.bruit_total()
    assert total == pytest.approx(2.5402e-6, rel=1e-9)
    assert total * 1e6 == pytest.approx(2.5402)
    assert abs(total * 1e6 - 2.504) > 0.03
    assert efficace * 1e3 == pytest.approx(1.594, abs=5e-4)
    assert bruit_efficace([2.504e-6]) * 1e3 == pytest.approx(1.58, abs=5e-3)
    assert densite * 1e6 == pytest.approx(11.27, abs=5e-3)
    assert densite == pytest.approx(densite_de_bruit_maximale(efficace, 20e3))


def test_ex11_l_etage_dominant() -> None:
    """Page 3 : le passe-haut, a lui seul les quatre cinquiemes du bruit."""
    nom, part = un.etage_dominant()
    assert nom == "passe-haut"
    assert part == pytest.approx(2e-6 / 2.5402e-6)


def test_ex11_le_bruit_rapporte_a_l_entree() -> None:
    """1,59 mV en sortie, soit 0,32 uV a l'entree : sous les 2 uV de resolution."""
    assert un.bruit_a_l_entree() == pytest.approx(math.sqrt(2.5402e-6) / 5000)
    assert un.bruit_a_l_entree() * 1e6 == pytest.approx(0.319, abs=5e-4)


# --- Exercice 1.2 -----------------------------------------------------


def test_ex12_le_gain_et_la_cadence() -> None:
    """Page 4 : 5000, et 40 kS/s par voie."""
    assert deux.gain() == pytest.approx(5000.0)
    assert deux.cadence_par_voie() == pytest.approx(40e3)
    assert deux.cadence_totale() == pytest.approx(320e3)
    assert deux.cadence_totale() != pytest.approx(360e3)


def test_ex12_le_convertisseur() -> None:
    """Pages 4 et 5 : onze bits au moins ; seul un ADC3 tient 320 kS/s."""
    assert deux.bits_minimaux() == 11
    assert bits_necessaires(2e-3, 1e-6) == 11
    assert bits_necessaires(1000.0, 1.0) == 10  # dix bits ne tiennent pas 2000 pas
    assert deux.convertisseurs_assez_fins() == ("ADC2", "ADC3")
    assert deux.cadence_disponible("ADC2") == pytest.approx(200e3)
    assert deux.cadence_disponible("ADC3") == pytest.approx(400e3)
    assert deux.cadence_disponible("ADC1") == pytest.approx(4e6)
    assert deux.convertisseurs_possibles() == ("ADC3",)
    with pytest.raises(KeyError, match="ADC9"):
        deux.cadence_disponible("ADC9")


def test_ex12_le_multiplexeur() -> None:
    """Page 5 : 0,32 uV^2 pour M2, sous le budget de 4 ; M2 est le moins cher."""
    assert pytest.approx(4e-12, rel=1e-9, abs=0) == UnDeux.BUDGET
    assert deux.bruit_du_multiplexeur("M2") == pytest.approx(0.32e-12, rel=1e-9, abs=0)
    assert deux.bruit_du_multiplexeur("M1") == pytest.approx(0.08e-12, rel=1e-9, abs=0)
    assert deux.multiplexeur_retenu() == "M2"


def test_ex12_ni_filtre_ni_multiplexeur_en_tete() -> None:
    """Page 5 : `2 10^6 uV^2` a l'entree, contre un budget de 4."""
    assert deux.bruit_d_un_bloc_en_tete(10e-6) == pytest.approx(2e6 * 1e-12)
    assert deux.bruit_d_un_bloc_en_tete(10e-6) > 1e5 * UnDeux.BUDGET


def test_ex12_le_bruit_en_courant_est_negligeable() -> None:
    """Page 5 : `2,048 10^-7 uV^2` pour le pire amplificateur."""
    assert deux.pire_bruit_en_courant() == pytest.approx(
        2.048e-7 * 1e-12, rel=1e-9, abs=0
    )
    assert deux.pire_bruit_en_courant() < 1e-6 * UnDeux.BUDGET


def test_ex12_le_premier_amplificateur() -> None:
    """Page 6 : 4, 25 et 0,25 uV^2 ; seul l'AMP3 laisse du budget, 3,75."""
    assert deux.bruits_des_amplificateurs() == pytest.approx(
        {"AMP1": 4e-12, "AMP2": 25e-12, "AMP3": 0.25e-12}, rel=1e-9, abs=0
    )
    assert deux.premiers_amplificateurs_possibles() == ("AMP3",)
    assert deux.budget_restant() == pytest.approx(3.75e-12, rel=1e-9, abs=0)
    assert deux.gain_maximal_du_premier() == pytest.approx(1250.0)


def test_ex12_le_deuxieme_bloc() -> None:
    """Page 6 : `1,6 10^-5` pour le pire ampli, 1,28 pour le filtre ; gain de 4."""
    assert budget.bruit_du_second_amplificateur("AMP2") == pytest.approx(
        1.6e-5 * 1e-12, rel=1e-9, abs=0
    )
    assert budget.bruit_du_filtre_en_second() == pytest.approx(
        1.28e-12, rel=1e-9, abs=0
    )
    assert budget.bruit_du_filtre_en_second() < deux.budget_restant()
    assert budget.gain_du_second() == pytest.approx(4.0)


def test_ex12_la_chaine_entiere_tient_le_budget() -> None:
    """AMP3, AMP1, filtre, M2, ADC3 : 0,65 uV^2 a l'entree, soit 0,81 uV."""
    puissances = budget.puissances_de_la_chaine()
    assert list(puissances) == ["AMP3", "AMP1", "filtre", "M2"]
    assert puissances["AMP3"] == pytest.approx(0.25e-12, rel=1e-9, abs=0)
    assert puissances["AMP1"] == pytest.approx(4e-12 / 1250**2, rel=1e-9, abs=0)
    assert puissances["filtre"] == pytest.approx(
        100e-12 * 20e3 / 5000**2, rel=1e-9, abs=0
    )
    assert puissances["M2"] == pytest.approx(0.32e-12, rel=1e-9, abs=0)
    assert budget.bruit_de_la_chaine() * 1e6 == pytest.approx(0.806, abs=5e-4)
    assert budget.bruit_de_la_chaine() ** 2 < UnDeux.BUDGET
    assert budget.cout() == pytest.approx(8 * 4 + 8 * 1 + 0.5)


# --- Exercice 1.3 -----------------------------------------------------


def test_ex13_le_convertisseur_et_les_multiplexeurs() -> None:
    """Page 7 : 12,29 donc 13 bits, l'ADC3 ; un multiplexeur de 32 et un de 16."""
    assert trois.bits_minimaux() == 13
    assert math.log2(5000) == pytest.approx(12.29, abs=5e-3)
    assert trois.convertisseurs_possibles() == ("ADC3",)
    assert trois.cadence_par_voie() == pytest.approx(1000.0)
    assert trois.multiplexeurs() == (32, 16)
    assert trois.cadence_du_plus_grand_multiplexeur() == pytest.approx(32e3)


def test_ex13_les_amplificateurs_alimentables() -> None:
    """Page 7 : l'AMP3 ne supporte pas 5 V ; l'AMP2 bruite trop."""
    assert trois.amplificateurs_alimentables() == ("AMP1", "AMP2", "AMP4")
    assert trois.premiers_amplificateurs_possibles() == ("AMP1", "AMP4")
    assert trois.amplificateur_retenu() == "AMP1"


def test_ex13_le_bruit_en_courant_et_le_filtre_en_tete() -> None:
    """Pages 7 et 8 : `2 10^-3 uV^2` de courant ; 22,36 uV pour un filtre en tete."""
    assert trois.pire_bruit_en_courant() == pytest.approx(2e-3 * 1e-12, rel=1e-9, abs=0)
    assert trois.bruit_efficace_en_tete(1e-6) == pytest.approx(22.36e-6, rel=1e-3)
    assert trois.bruit_efficace_en_tete(1e-6) > 5e-6


def test_ex13_les_gains() -> None:
    """Pages 8 et 9 : 20 avant le passe-haut, puis 50."""
    assert trois.gains() == pytest.approx((20.0, 50.0))
    assert trois.produit_gain_bande_requis() == pytest.approx(10e3)


def test_ex13_le_budget_bloc_par_bloc() -> None:
    """Pages 8 et 9 : 25, puis 24,86, 23,61, 23,61, 23,56 uV^2."""
    puissances = trois.puissances()
    assert list(puissances) == ["AMP1", "passe-haut", "AMP1 bis", "passe-bas", "MUX"]
    attendues = [0.375**2, 1.25, 0.375**2 / 400, 0.05, 0.05]
    assert [p * 1e12 for p in puissances.values()] == pytest.approx(attendues)
    budgets = [b * 1e12 for b in trois.budgets()]
    assert budgets[:5] == pytest.approx([25.0, 24.86, 23.61, 23.61, 23.56], abs=5e-3)
    assert puissances["AMP1 bis"] * 1e12 == pytest.approx(3.5156e-4, rel=1e-4)


def test_ex13_un_passe_bas_en_deuxieme_position_est_exclu() -> None:
    """Page 8 : 125 uV^2, cent fois le passe-haut, cinq fois le budget."""
    assert trois.bruit_du_passe_bas_en_second() == pytest.approx(
        125e-12, rel=1e-9, abs=0
    )
    assert trois.bruit_du_passe_bas_en_second() > trois.budgets()[1]


def test_ex13_le_total_tient_largement() -> None:
    """Ce que l'officiel ne somme pas : 1,49 uV^2, soit 1,22 uV pour 5 permis."""
    assert trois.bruit_total() * 1e6 == pytest.approx(1.221, abs=5e-4)
    assert trois.bruit_total() < 5e-6


def test_ex13_un_seul_convertisseur_suffirait() -> None:
    """45 voies a 1 kS/s font 45 kS/s : sous les 50 de l'ADC3."""
    assert trois.cadence_totale() == pytest.approx(45e3)
    assert trois.cadence_totale() <= UnTrois.CONVERTISSEURS["ADC3"][1]
    assert trois.bruit_total(multiplexeurs_en_cascade=2) * 1e6 == pytest.approx(
        math.sqrt(1.4909765625 + 0.05), abs=5e-4
    )


# --- Exercice 1.4 -----------------------------------------------------


def test_ex14_les_convertisseurs_et_les_multiplexeurs() -> None:
    """Page 11 : 40 kHz a tenir, donc deux CAN a 30 kHz et deux MUX de 128 voies."""
    assert quatre.cadence_totale() == pytest.approx(40e3)
    assert quatre.nombre_de_convertisseurs() == 2
    assert quatre.voies_par_convertisseur() == 100
    assert quatre.multiplexeur() == "MUX3"
    assert quatre.multiplexeur(voies=40) == "MUX2"
    assert quatre.multiplexeur(voies=32) == "MUX1"
    assert quatre.cadence_reelle_par_voie() == pytest.approx(300.0)
    assert quatre.bande_d_arret() == pytest.approx(150.0)


def test_ex14_le_filtre_ne_peut_pas_etre_en_tete() -> None:
    """Page 12 : 2 mV par racine de hertz, soit 20 mV efficaces contre 60 uV."""
    assert quatre.densite_du_filtre_en_tete() == pytest.approx(2e-3, rel=1e-5)
    assert quatre.bruit_du_filtre_en_tete() == pytest.approx(20e-3, rel=1e-5)
    assert quatre.bruit_du_filtre_en_tete() > 300 * UnQuatre.BUDGET


def test_ex14_les_amplificateurs() -> None:
    """Page 12 : 3 uV, 30 uV et 3 mV par racine de hertz ; seul l'AMP1 passe."""
    densites = quatre.densites_des_amplificateurs()
    assert [d * 1e6 for d in densites.values()] == pytest.approx(
        [3.0, 30.0, 3000.0], rel=1e-4
    )
    assert quatre.bruits_des_amplificateurs()["AMP1"] == pytest.approx(30e-6, rel=1e-4)
    assert quatre.amplificateurs_possibles() == ("AMP1",)
    assert quatre.gain() == pytest.approx(500.0)
    assert quatre.produit_gain_bande_requis() == pytest.approx(50e3)


def test_ex14_le_convertisseur() -> None:
    """Page 13 : cent niveaux, donc sept bits ; a 30 kHz, les CAN 4, 5 et 6."""
    assert quatre.bits_minimaux() == 7
    assert quatre.convertisseurs_possibles() == ("CAN4", "CAN5", "CAN6")


def test_ex14_le_bruit_total() -> None:
    """Pages 13 et 14 : 1, 30, 0,04, 0,2 et 0,4 uV ; 30 uV en tout."""
    efficaces = quatre.bruits_a_l_entree()
    assert list(efficaces) == ["capteur", "AMP1", "passe-bas", "MUX3", "CAN"]
    assert [e * 1e6 for e in efficaces.values()] == pytest.approx(
        [1.0, 30.0, 0.04, 0.2, 0.4], rel=1e-4
    )
    assert quatre.bruit_total() * 1e6 == pytest.approx(30.02, abs=5e-3)
    assert quatre.bruit_total() < UnQuatre.BUDGET


def test_ex14_la_reserve_sur_les_bruits_non_filtres() -> None:
    """Le MUX et le CAN suivent le filtre : sur 15 kHz, le total reste a 30,5 uV."""
    large = quatre.bruit_total(bande_apres_le_filtre=15e3)
    assert large * 1e6 == pytest.approx(30.51, abs=5e-3)
    assert large < UnQuatre.BUDGET


def test_ex14_le_nombre_de_composants() -> None:
    """Deux cents amplificateurs, deux cents filtres, deux MUX, deux CAN."""
    assert quatre.composants() == {"AMP1": 200, "passe-bas": 200, "MUX3": 2, "CAN": 2}
