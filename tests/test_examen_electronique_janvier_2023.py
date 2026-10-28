"""Les reponses du corrige manuscrit de janvier 2022-2023, recalculees.

La transcription de
`ElectroniqueInstrumentation/EXAMENS/CorrigeElecInstru202223_S1.pdf` porte
dix questions et quatre ecarts. Ces tests refont chaque calcul depuis les
donnees de l'enonce, sans reprendre le chemin du manuscrit.
"""

import math

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    courant_de_bruit_integre,
    densite_de_bruit_maximale,
    gain_de_chaine,
    nyquist,
    puissance_de_bruit_en_courant,
    tension_de_bruit_en_courant,
)
from polytech.electronique.domain.amplification import nombre_d_etages
from polytech.electronique.domain.chebyshev import elements_de_chebyshev
from polytech.electronique.domain.filtre import PrototypeDeChebyshev
from polytech.electronique.domain.induction import BoucleRectangulaire
from polytech.electronique.domain.laplace import reponse_a_un_echelon
from polytech.electronique.domain.pole_zero import pente_asymptotique
from polytech.electronique.domain.quadripole import (
    admittance_d_entree_a_vide,
    matrice_y_du_circuit,
)
from polytech.electronique.domain.retroaction import parametres_h_du_circuit

NANOHENRY = 1e-9
PICOFARAD = 1e-12
MILLIFARAD = 1e-3
MICROVOLT = 1e-6
NANOVOLT = 1e-9
FEMTOAMPERE = 1e-15

# --- Question 1 : l'integrateur et son courant de polarisation -----------


def test_le_montage_est_un_inverseur_de_gain_continu_moins_rf_sur_r() -> None:
    """Question 1.1 : `-Z_2/Z_1` avec `Z_2 = R_f // 1/(sC_f)`."""
    s, r, rf, cf = sp.symbols("s R R_f C_f", positive=True)
    impedance = rf / (s * cf * rf + 1)
    transfert = -impedance / r
    assert sp.simplify(transfert.subs(s, 0) + rf / r) == 0


def test_a_contre_reaction_infinie_le_montage_devient_un_integrateur() -> None:
    """Question 1.1 : `R_f -> infini` laisse `-1/(s C_f R)`."""
    s, r, rf, cf = sp.symbols("s R R_f C_f", positive=True)
    transfert = -(rf / (s * cf * rf + 1)) / r
    assert sp.simplify(sp.limit(transfert, rf, sp.oo) + 1 / (s * cf * r)) == 0


def test_le_transfert_du_courant_de_polarisation_a_la_forme_annoncee() -> None:
    """Question 1.2 : `A/(s + B)` avec `A = 1/C_f` et `B = 1/(R_f C_f)`."""
    s, rf, cf = sp.symbols("s R_f C_f", positive=True)
    transfert = rf / (s * cf * rf + 1)
    attendu = (1 / cf) / (s + 1 / (cf * rf))
    assert sp.simplify(transfert - attendu) == 0


def test_la_reponse_au_courant_en_echelon_est_celle_du_corrige() -> None:
    """Question 1.3 : `R_f I_0 (1 - exp(-t/(R_f C_f)))`."""
    s, t, rf, cf, i0 = sp.symbols("s t R_f C_f I_0", positive=True)
    obtenue = reponse_a_un_echelon((1 / cf) / (s + 1 / (cf * rf)), i0, s, t)
    attendue = rf * i0 * (1 - sp.exp(-t / (rf * cf)))
    assert sp.simplify(obtenue - attendue) == 0


def test_la_sortie_part_de_zero_et_arrive_a_rf_i0() -> None:
    """Question 1.3 : les deux verifications que l'enonce demande."""
    t, rf, cf, i0 = sp.symbols("t R_f C_f I_0", positive=True)
    sortie = rf * i0 * (1 - sp.exp(-t / (rf * cf)))
    assert sp.limit(sortie, t, 0, "+") == 0
    assert sp.simplify(sp.limit(sortie, t, sp.oo) - rf * i0) == 0


# --- Question 2 : les deux 2-ports en parallele --------------------------


def test_les_parametres_y_du_deux_ports_du_bas_sont_ceux_de_l_enonce() -> None:
    """Question 2.1 : `y_11 = 1/R_1`, `y_12 = 0`, `y_21 = g_m`, `y_22 = 1/R_2`."""
    r1, r2, gm = sp.symbols("R_1 R_2 g_m", positive=True)
    v1, v2 = sp.symbols("V_1 V_2")
    y = matrice_y_du_circuit(v1 / r1, gm * v1 + v2 / r2, v1, v2)
    attendue = sp.Matrix([[1 / r1, 0], [gm, 1 / r2]])
    assert sp.simplify(y - attendue) == sp.zeros(2, 2)


def test_les_parametres_h_du_deux_ports_du_bas_sont_ceux_du_corrige() -> None:
    """Question 2.1 : `h_11 = R_1`, `h_21 = g_m R_1`, `h_22 = 1/R_2`."""
    r1, r2, gm = sp.symbols("R_1 R_2 g_m", positive=True)
    i1, v2 = sp.symbols("I_1 V_2")
    tension = r1 * i1
    h = parametres_h_du_circuit(tension, gm * tension + v2 / r2, i1, v2)
    attendue = sp.Matrix([[r1, 0], [gm * r1, 1 / r2]])
    assert sp.simplify(h - attendue) == sp.zeros(2, 2)


def test_les_parametres_du_deux_ports_du_haut_sont_ceux_de_l_enonce() -> None:
    """Question 2.1 : la seule `R_f`, vue en y puis en h."""
    rf = sp.Symbol("R_f", positive=True)
    v1, v2 = sp.symbols("V_1 V_2")
    y = matrice_y_du_circuit((v1 - v2) / rf, (v2 - v1) / rf, v1, v2)
    assert sp.simplify(y - sp.Matrix([[1 / rf, -1 / rf], [-1 / rf, 1 / rf]])) == (
        sp.zeros(2, 2)
    )
    i1 = sp.Symbol("I_1")
    h = parametres_h_du_circuit(rf * i1 + v2, -i1, i1, v2)
    assert sp.simplify(h - sp.Matrix([[rf, 1], [-1, 0]])) == sp.zeros(2, 2)


def test_l_admittance_d_entree_a_vide_est_celle_du_corrige() -> None:
    """Question 2.3 : `y_11 - y_12 y_21 / y_22`."""
    y11, y12, y21, y22 = sp.symbols("y_11 y_12 y_21 y_22")
    matrice = sp.Matrix([[y11, y12], [y21, y22]])
    obtenue = admittance_d_entree_a_vide(matrice)
    assert sp.simplify(obtenue - (y11 - y12 * y21 / y22)) == 0


# --- Questions 3 et 4 : lecture et cascade -------------------------------


def test_cinq_poles_et_quatre_zeros_donnent_moins_vingt_decibels() -> None:
    """Question 3.2."""
    assert pente_asymptotique(zeros=4, poles=5) == pytest.approx(-20.0)


def test_trois_etages_sont_necessaires_pour_un_gain_de_cinquante() -> None:
    """Question 4.1 : 2 MHz sur 500 kHz plafonne a 4 par etage."""
    assert nombre_d_etages(50.0, 2e6, 500e3) == 3
    assert 4**2 < 50
    assert 4**3 >= 50


# --- Question 6 : le passe-haut de Chebyshev -----------------------------


def test_la_table_de_chebyshev_donne_1_1474_et_non_1_1979() -> None:
    """Question 6.1 : le prototype se calcule, et le corrige l'a recopie.

    Les formules fermees rendent `(1,0316 ; 1,1474 ; 1,0316)`. Le corrige
    ecrit le premier et le troisieme exactement, et `1,1979` pour le
    second -- mais la suite de son calcul emploie bien `1,1474`, puisque
    `1,1474/50` vaut les `22,95 mF` qu'il annonce.
    """
    g1, g2, g3 = elements_de_chebyshev(0.1, 3)
    assert g1 == pytest.approx(1.0316, rel=1e-4)
    assert g2 == pytest.approx(1.1474, rel=1e-4)
    assert g3 == pytest.approx(g1)
    assert g2 != pytest.approx(1.1979, rel=1e-3)
    assert g2 / 50 == pytest.approx(22.948 * MILLIFARAD, rel=1e-4)


def test_le_passe_haut_de_chebyshev_donne_7_7_pf_et_17_3_nh() -> None:
    """Question 6.1 : prototype 0,1 dB d'ordre 3, 50 ohms, 400 MHz."""
    prototype = PrototypeDeChebyshev(
        elements=elements_de_chebyshev(0.1, 3),
        impedance=50.0,
    )
    c1, l2, c3 = prototype.en_passe_haut(2 * math.pi * 400e6)
    assert c1 == pytest.approx(7.72 * PICOFARAD, rel=1e-2, abs=0)
    assert l2 == pytest.approx(17.34 * NANOHENRY, rel=1e-2)
    assert c3 == pytest.approx(c1, rel=1e-9, abs=0)


def test_les_17_4_millihenrys_du_corrige_sont_des_nanohenrys() -> None:
    """Question 6.1 : le chiffre est juste, l'unite est six ordres trop haut."""
    prototype = PrototypeDeChebyshev(
        elements=elements_de_chebyshev(0.1, 3),
        impedance=50.0,
    )
    _, l2, _ = prototype.en_passe_haut(2 * math.pi * 400e6)
    assert l2 / 17.4e-3 == pytest.approx(1e-6, rel=1e-2)


# --- Question 9 : la boucle coupee en deux -------------------------------


def test_la_tension_aux_bornes_de_r_est_celle_du_corrige() -> None:
    """Question 9.1 : `-a b B_0 omega cos(omega t)`."""
    a, b, champ, omega, t = sp.symbols("a b B_0 omega t", positive=True)
    boucle = BoucleRectangulaire(longueur=a, largeur=b)
    obtenue = boucle.tension_induite(champ, omega, t)
    attendue = -a * b * champ * omega * sp.cos(omega * t)
    assert sp.simplify(obtenue - attendue) == 0


def test_le_voltmetre_qui_coupe_au_milieu_ne_lit_que_la_moitie() -> None:
    """Question 9.2 : la moitie de la surface, donc la moitie de la tension."""
    a, b, champ, omega, t = sp.symbols("a b B_0 omega t", positive=True)
    entiere = BoucleRectangulaire(longueur=a, largeur=b)
    moitie = BoucleRectangulaire(longueur=a, largeur=b / 2)
    rapport = moitie.tension_induite(champ, omega, t) / entiere.tension_induite(
        champ,
        omega,
        t,
    )
    assert sp.simplify(rapport - sp.Rational(1, 2)) == 0


# --- Question 10 : trente-deux capteurs de lumiere -----------------------


def test_le_gain_global_de_la_chaine_vaut_12500() -> None:
    """Question 10.1 : porter 0,4 mV a 5 V."""
    assert gain_de_chaine(5.0, 0.4e-3) == pytest.approx(12500.0)


def test_chaque_signal_demande_cinq_kilohertz() -> None:
    """Question 10.1 : Nyquist sur 2,5 kHz."""
    assert nyquist(2500.0) == pytest.approx(5000.0)


def test_onze_bits_sont_necessaires_donc_seul_l_adc3_convient() -> None:
    """Question 10.3 : 800 microvolts par pas d'un demi-microvolt."""
    assert bits_necessaires(800 * MICROVOLT, 0.5 * MICROVOLT) == 11


def test_seize_voies_a_cinq_kilohertz_tiennent_dans_cent_kilo_echantillons() -> None:
    """Question 10.3 : d'ou deux ADC3 pour trente-deux capteurs."""
    assert cadence_du_multiplexeur(5000.0, voies=16) == pytest.approx(80e3)
    assert cadence_du_multiplexeur(5000.0, voies=16) < 100e3


def test_le_multiplexeur_en_tete_deposerait_288_microvolts_de_bruit() -> None:
    """Question 10.4 : 6 microvolts par racine de hertz sur 2300 Hz."""
    bruit = 6 * MICROVOLT * math.sqrt(2500 - 200)
    assert bruit == pytest.approx(287.75 * MICROVOLT, rel=1e-3)
    assert bruit > 1 * MICROVOLT


def test_le_filtre_en_tete_en_deposerait_48() -> None:
    """Question 10.5 : un microvolt par racine de hertz sur la meme bande."""
    bruit = courant_de_bruit_integre(1 * MICROVOLT, 2500.0 - 200.0)
    assert bruit == pytest.approx(47.96 * MICROVOLT, rel=1e-3)
    assert bruit > 1 * MICROVOLT


def test_le_pire_bruit_en_courant_vaut_9592_femtoamperes() -> None:
    """Question 10.6 : 200 fA par racine de hertz sur 2300 Hz."""
    assert courant_de_bruit_integre(200 * FEMTOAMPERE, 2300.0) == pytest.approx(
        9592 * FEMTOAMPERE, rel=1e-4, abs=0
    )


def test_ce_bruit_en_courant_fait_1_92_nanovolt_et_non_millivolt() -> None:
    """Question 10.6 : `3,68e-18 V^2` fait 3,68 nanovolts carres.

    Le corrige ecrit « 3,68 (mV)^2 » puis « 1,92 mV << 1 microV », ce que
    l'inegalite elle-meme contredit.
    """
    puissance = puissance_de_bruit_en_courant(200 * FEMTOAMPERE, 200.0, 2300.0)
    tension = tension_de_bruit_en_courant(200 * FEMTOAMPERE, 200.0, 2300.0)
    assert tension**2 == pytest.approx(puissance, rel=1e-12)
    assert puissance == pytest.approx(3.68e-18, rel=1e-9, abs=0)
    assert tension == pytest.approx(1.92 * NANOVOLT, abs=0.005 * NANOVOLT)
    assert tension < 1 * MICROVOLT


def test_la_densite_maximale_admissible_vaut_20_85_nanovolts() -> None:
    """Question 10.6 : c'est elle qui ecarte AMP1 et garde AMP2 et AMP3."""
    plafond = densite_de_bruit_maximale(1 * MICROVOLT, 2300.0)
    assert plafond == pytest.approx(20.85 * NANOVOLT, rel=1e-3)
    assert plafond < 200 * NANOVOLT
    assert plafond > 10 * NANOVOLT
    assert plafond > 4 * NANOVOLT


def test_le_gain_du_premier_etage_ne_peut_depasser_125() -> None:
    """Question 10.7 : la saturation, pas le produit gain-bande, commande."""
    assert pytest.approx(125.0) == 5000.0 / (15.0 + 25.0)
    assert 125 * 2500 < 4e6
