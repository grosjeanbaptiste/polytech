"""Le syllabus d'electronique, chapitres 3 et 5 : ce qu'il ajoute aux planches.

Source : `Syllabus_Electronics_and_Instrumentation.pdf`, dans
`ElectroniqueInstrumentation/SYLLABUS/`, sections 3.9, 3.10.1, 5.4.2,
5.10 a 5.13 (pages 138 a 147 et 211 a 245), lues sur leur couche de texte.

Les oracles : l'exemple 3-1 de l'amplificateur audio, la derivation de
l'equation (3.23) de la paire differentielle, la loi de Poisson (5.5),
l'exemple 5-2 de l'ordre des amplificateurs, les figures 5.14 a 5.16 du
convertisseur quatre bits, l'echantillonneur-bloqueur de la figure 5.18
et le dimensionnement de la page 241.

Quatre ecarts.

* Exemple 3-1 : le gain de 80 y est « split in two parts: 40 for each
  stage ». Les gains d'une cascade se multiplient -- la section 3.6.3 le
  dit elle-meme -- : deux etages de 40 font 1600. Et leurs deux bandes de
  25 kHz, 24,4 pour l'inverseur, donnent en cascade 16 kHz, non les 20
  demandes. Un partage en 8,9 par etage tient les deux exigences.
* Page 140 : la capacite d'entree de 4 pF aurait pour impedance « 250
  kohm at 1 MHz » ; c'est 39,8 kohm, le `2 pi` y manque. La meme page
  ecrit que le gain du CA3140 « is still approximately equal to 1e5 » a
  1 MHz : avec 4,5 MHz de produit gain-bande, il y vaut 4,5.
* Page 232 : la condition de repliement y est ecrite de 0 a `f_s`, puis de
  `f_s` a l'infini ; le spectre se replie autour de `f_s / 2`, comme
  l'integrale de la planche 18 du chapitre 4 l'ecrit.
* Pages 144 et 147 : la capacite de compensation du CA3140 vaut 12 pF sur
  son schema, « 16 pF » dans le texte de la section 3.10.2.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import gain_de_chaine
from polytech.electronique.domain.amplification import (
    gain_maximal_par_etage,
    nombre_d_etages,
)
from polytech.electronique.domain.aop_interne import (
    bande_en_boucle_ouverte,
    courant_de_la_paire_differentielle,
    courants_de_la_paire,
    module_du_gain_ouvert,
    vitesse_de_balayage,
)
from polytech.electronique.domain.boucle_fermee import (
    gain_vu_de_la_charge,
    gain_vu_de_la_source,
)
from polytech.electronique.domain.cascade_d_etages import (
    bande_de_la_cascade,
    nombre_d_etages_en_cascade,
)
from polytech.electronique.domain.echantillonneur_bloqueur import (
    constante_de_temps,
    fraction_perdue,
)
from polytech.electronique.domain.friis import bruit_ramene_a_l_entree
from polytech.electronique.domain.multiplexage import (
    attenuation_de_la_resistance_passante,
    demultiplexer,
)
from polytech.electronique.domain.quantification import Quantificateur
from polytech.maths.domain.probabilites.loi_de_poisson import (
    probabilite_de_poisson,
)
from polytech.maths.domain.probabilites.loi_normale import densite_normale

PRODUIT, BANDE = 1e6, 20e3
"""Exemple 3-1 : 1 MHz de produit gain-bande, 20 kHz de bande audio."""


def test_l_exemple_3_1_le_gain_maximal_par_etage() -> None:
    """Page 141 : `1 MHz / 20 kHz = 50`, donc deux etages pour un gain de 80."""
    assert gain_maximal_par_etage(PRODUIT, BANDE) == 50.0
    assert nombre_d_etages(80.0, PRODUIT, BANDE) == 2
    assert 1 + 39e3 / 1e3 == 40
    assert 80e3 / 2e3 == 40


def test_l_exemple_3_1_tel_qu_imprime_ne_tient_ni_son_gain_ni_sa_bande() -> None:
    """Deux etages de 40 : un gain de 1600, et deux poles a 25 kHz qui coupent a 16."""
    assert 40 * 40 == 1600
    par_etage = PRODUIT / 40
    assert par_etage == 25e3
    cascade = bande_de_la_cascade(2, par_etage)
    assert cascade == pytest.approx(16.1e3, rel=3e-3)
    assert cascade < BANDE
    inverseur = PRODUIT / (1 + 40)
    carres = inverseur**2, par_etage**2
    somme, produit = sum(carres), math.prod(carres)
    exacte = math.sqrt((-somme + math.sqrt(somme**2 + 4 * produit)) / 2)
    filtres = math.hypot(1, exacte / inverseur) * math.hypot(1, exacte / par_etage)
    assert 1 / filtres == pytest.approx(1 / math.sqrt(2), rel=1e-9)
    assert exacte == pytest.approx(15.9e3, rel=3e-3)


def test_l_exemple_3_1_se_tient_avec_un_partage_en_racine() -> None:
    """Deux etages de `racine(80) = 8,9` : 112 kHz chacun, 72 kHz en cascade."""
    assert nombre_d_etages_en_cascade(80.0, PRODUIT, BANDE) == 2
    par_etage = math.sqrt(80.0)
    assert par_etage**2 == pytest.approx(80.0)
    bande = bande_de_la_cascade(2, PRODUIT / par_etage)
    assert bande == pytest.approx(72e3, rel=3e-3)
    assert bande > BANDE


def test_l_exemple_3_1_les_impedances() -> None:
    """Page 142 : 98,4 % a l'entree, 0,6 ohm de sortie, 250 Mohm, 97 % sur la charge."""
    assert gain_vu_de_la_source(1.0, 32.0, 2000.0) == pytest.approx(0.984, abs=5e-4)
    sortie = 150.0 * 40 / 1e4
    assert sortie == pytest.approx(0.6)
    assert 1e6 * 1e4 / 40 == 250e6
    assert gain_vu_de_la_charge(1.0, sortie, 20.0) == pytest.approx(0.971, abs=5e-4)


def test_la_bande_ouverte_du_ca3140_au_pire() -> None:
    """Page 139 : 45 Hz au gain typique de 1e5, 225 Hz au gain minimal de 2e4."""
    assert bande_en_boucle_ouverte(4.5e6, 1e5) == pytest.approx(45.0)
    assert bande_en_boucle_ouverte(4.5e6, 2e4) == pytest.approx(225.0)


def test_la_page_140_oublie_deux_pi_et_le_pole() -> None:
    """A 1 MHz : 39,8 kohm et non 250 ; un gain de 4,5 et non de 1e5."""
    a_un_megahertz = 1 / (2 * math.pi * 1e6 * 4e-12)
    assert a_un_megahertz == pytest.approx(39.8e3, rel=1e-3)
    assert pytest.approx(250e3) == 1 / (1e6 * 4e-12)
    gain = module_du_gain_ouvert(1e6, produit=4.5e6, continu=1e5)
    assert gain == pytest.approx(4.5, rel=1e-6)
    assert module_du_gain_ouvert(1.0, produit=4.5e6, continu=1e5) == pytest.approx(
        1e5, rel=1e-3
    )
    encore_dominant = gain * a_un_megahertz / 100.0
    assert encore_dominant / 0.01 > 1e5
    with pytest.raises(ValueError, match="frequence negative"):
        module_du_gain_ouvert(-1.0, produit=4.5e6, continu=1e5)


def test_la_derivation_de_la_paire_differentielle() -> None:
    """Pages 146 et 147 : `I_l / I_1 = 1 / (1 + e^-x)`, `I_r / I_1 = 1 / (1 + e^x)`."""
    queue, thermique = 2e-4, 25e-3
    for reduit in (-3.0, -0.5, 0.0, 1.5):
        tension = reduit * thermique
        gauche, droite = courants_de_la_paire(tension, queue, thermique=thermique)
        assert gauche == pytest.approx(queue / (1 + math.exp(-reduit)))
        assert droite == pytest.approx(queue / (1 + math.exp(reduit)))
        assert gauche + droite == pytest.approx(queue)
        difference = courant_de_la_paire_differentielle(
            tension, queue, thermique=thermique
        )
        assert gauche - droite == pytest.approx(difference, abs=1e-18)
    au_repos = courants_de_la_paire(0.0, queue, thermique=thermique)
    assert au_repos == pytest.approx((queue / 2, queue / 2))


def test_la_vitesse_de_balayage_du_schema_du_ca3140() -> None:
    """Page 144 : 200 uA dans 12 pF ; le texte de la section 3.10.2 ecrit 16 pF."""
    schema = vitesse_de_balayage(200e-6, 12e-12)
    texte = vitesse_de_balayage(200e-6, 16e-12)
    assert schema == pytest.approx(16.7e6, rel=2e-3)
    assert texte == pytest.approx(12.5e6)


def test_la_loi_de_poisson_tend_vers_la_loi_normale() -> None:
    """(5.5) : `lambda^k e^-lambda / k!` ; normale des que `lambda` atteint 1000."""
    assert probabilite_de_poisson(0, 2.0) == pytest.approx(math.exp(-2.0))
    assert probabilite_de_poisson(3, 2.0) == pytest.approx(8 * math.exp(-2.0) / 6)
    total = sum(probabilite_de_poisson(k, 4.0) for k in range(60))
    assert total == pytest.approx(1.0)
    moyenne = sum(k * probabilite_de_poisson(k, 4.0) for k in range(60))
    assert moyenne == pytest.approx(4.0)
    for rang in (1000, 1030, 950):
        normale = float(densite_normale(sp.Integer(rang), 1000, 1000))
        poisson = probabilite_de_poisson(rang, 1000.0)
        assert poisson == pytest.approx(normale, rel=0.03)
    with pytest.raises(ValueError, match="rang negatif"):
        probabilite_de_poisson(-1, 2.0)
    with pytest.raises(ValueError, match="parametre nul ou negatif"):
        probabilite_de_poisson(1, 0.0)


def test_l_exemple_5_2_l_ordre_des_amplificateurs() -> None:
    """Page 228 : 1e5 et 1e6 nV^2, gains de 10 ; 110 000 contre 1 001 000 nV^2."""
    faible_bruit, general = 100.0 * 1e3, 1000.0 * 1e3
    assert (faible_bruit, general) == (1e5, 1e6)
    bon = bruit_ramene_a_l_entree(((faible_bruit, 10.0), (general, 10.0)))
    mauvais = bruit_ramene_a_l_entree(((general, 10.0), (faible_bruit, 10.0)))
    assert bon == pytest.approx(110_000.0)
    assert mauvais == pytest.approx(1_001_000.0)
    assert mauvais / bon == pytest.approx(9.1, abs=0.01)
    trois = ((faible_bruit, 10.0), (general, 10.0), (1e7, 10.0))
    assert bruit_ramene_a_l_entree(trois) == pytest.approx(1e5 + 1e4 + 1e3)
    with pytest.raises(ValueError, match="cascade vide"):
        bruit_ramene_a_l_entree(())


QUATRE_BITS = Quantificateur(reference=5.0, bits=4)


def test_les_figures_5_14_a_5_16_du_convertisseur_quatre_bits() -> None:
    """Pages 232 a 234 : l'erreur a 5 ms, 1 V qui gache les codes, 8 V qui sature."""
    a_cinq_millisecondes = 5.0 * math.sin(2 * math.pi * 100 * 5e-3)
    assert a_cinq_millisecondes == pytest.approx(0.0, abs=1e-12)
    niveau = QUATRE_BITS.niveau(QUATRE_BITS.code(0.0))
    assert abs(niveau) == pytest.approx(QUATRE_BITS.pas() / 2)
    instants = [k / 2000 for k in range(40)]
    faibles = {QUATRE_BITS.code(math.sin(2 * math.pi * 100 * t)) for t in instants}
    assert faibles == {6, 7, 8, 9}
    satures = [QUATRE_BITS.code(8 * math.sin(2 * math.pi * 100 * t)) for t in instants]
    assert satures.count(15) > 8
    assert QUATRE_BITS.code(5.0) == QUATRE_BITS.code(30.0)


def test_l_echantillonneur_bloqueur_de_la_figure_5_18() -> None:
    """Page 235 : charge par `C R_o`, maintien par `C R_in` ; 1 ohm et 1 Gohm."""
    capacite = 1e-9
    charge = constante_de_temps(capacite, 1.0)
    maintien = constante_de_temps(capacite, 1e9)
    assert charge == pytest.approx(1e-9)
    assert maintien == pytest.approx(1.0)
    assert maintien / charge == pytest.approx(1e9)
    assert fraction_perdue(1e-3, capacite, 1e9) == pytest.approx(1e-3, rel=1e-3)
    assert fraction_perdue(5 * charge, capacite, 1.0) == pytest.approx(1 - math.exp(-5))
    with pytest.raises(ValueError, match="capacite ou resistance nulle ou negative"):
        constante_de_temps(0.0, 1.0)
    with pytest.raises(ValueError, match="duree negative"):
        fraction_perdue(-1.0, capacite, 1.0)


def test_le_demultiplexage_de_la_page_238() -> None:
    """`v_1[k] = v_ADC[2 k]` et `v_2[k] = v_ADC[2 k + 1]`."""
    melange = (10, 20, 11, 21, 12, 22)
    assert demultiplexer(melange, 2) == ((10, 11, 12), (20, 21, 22))
    assert demultiplexer(melange, 3) == ((10, 21), (20, 12), (11, 22))
    with pytest.raises(ValueError, match="nombre d'echantillons non multiple"):
        demultiplexer(melange, 4)


def test_les_imperfections_du_multiplexeur() -> None:
    """Pages 238 et 239 : 100 ohm passants pour un DG1208, 2 ohm pour un TMUX7219."""
    charge = 10e3
    dg1208 = attenuation_de_la_resistance_passante(100.0, charge)
    tmux7219 = attenuation_de_la_resistance_passante(2.0, charge)
    assert dg1208 == pytest.approx(0.990, abs=5e-4)
    assert tmux7219 == pytest.approx(0.9998, abs=5e-5)
    assert pytest.approx(10.5e-9) == 300.0 * 35e-12
    assert 150e-9 / (300.0 * 35e-12) > 14
    with pytest.raises(ValueError, match="resistance negative ou charge nulle"):
        attenuation_de_la_resistance_passante(100.0, 0.0)


def test_le_dimensionnement_des_gains_de_la_page_241() -> None:
    """50 uV sur 100 mV, convertisseur sur +-5 V : moins de 50, puis moins de 2000."""
    premier = gain_de_chaine(5.0, 100e-3 + 50e-6)
    assert premier < 50
    assert premier == pytest.approx(49.975, abs=1e-3)
    apres_filtrage = 50 * 50e-6
    assert apres_filtrage == pytest.approx(2.5e-3)
    assert gain_de_chaine(5.0, apres_filtrage) == pytest.approx(2000.0)
    assert pytest.approx(5.0) == 50 * 2000 * 50e-6
