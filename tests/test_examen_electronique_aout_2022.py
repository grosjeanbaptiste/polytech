"""Les reponses du corrige manuscrit d'aout 2021-2022, recalculees.

La transcription de
`ElectroniqueInstrumentation/EXAMENS/ExamenS2_202122Corrige.pdf` porte neuf
questions et deux ecarts. Ces tests refont chaque calcul depuis les donnees
de l'enonce, sans reprendre le chemin du manuscrit.
"""

import math

import pytest
import sympy as sp

from polytech.electronique.application import chaine_1_2_lab_03 as chaine_1_2
from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    gain_de_chaine,
    nyquist,
    puissance_de_bruit_en_courant,
)
from polytech.electronique.domain.amplification import (
    gain_maximal_par_etage,
    gain_non_inverseur,
)
from polytech.electronique.domain.bruit import densite_spectrale_unilaterale
from polytech.electronique.domain.filtre import PrototypeButterworth
from polytech.electronique.domain.friis import facteur_d_une_cascade
from polytech.electronique.domain.nodale import transfert_avec_transconductance

KILOHM = 1e3
MEGOHM = 1e6
NANOHENRY = 1e-9
PICOFARAD = 1e-12
MICROVOLT = 1e-6

# --- Question 1 : l'etage a transconductance ------------------------------


def test_le_transfert_exact_tend_vers_l_approximation_du_corrige() -> None:
    """Question 1.2 : « sans inverser le systeme » revient a ouvrir `R_f`.

    Le corrige decoupe le circuit en deux morceaux independants, ce qui
    suppose que `R_f` ne charge pas le premier. Le systeme complet le
    confirme : son transfert tend vers cette expression quand `R_f` part a
    l'infini.
    """
    rs, rin, rf, rc, gm = sp.symbols("R_S R_in R_f R_C g_m", positive=True)
    exact = transfert_avec_transconductance(rs, rin, rf, rc, gm)
    approche = -gm * rc * rin / (rs + rin)
    assert sp.simplify(sp.limit(exact, rf, sp.oo) - approche) == 0


def test_le_transfert_exact_n_est_pas_l_approximation() -> None:
    """Question 1.2 : l'approximation est une approximation, et cela se mesure."""
    rs, rin, rf, rc, gm = sp.symbols("R_S R_in R_f R_C g_m", positive=True)
    exact = transfert_avec_transconductance(rs, rin, rf, rc, gm)
    approche = -gm * rc * rin / (rs + rin)
    assert sp.simplify(exact - approche) != 0


# --- Questions 2 et 3 : deux filtres et leur bruit ------------------------


def test_la_cellule_rc_a_le_transfert_annonce() -> None:
    """Question 2.1 : diviseur de tension."""
    s, r, c = sp.symbols("s R C", positive=True)
    divise = (1 / (s * c)) / (r + 1 / (s * c))
    assert sp.simplify(divise - 1 / (s * r * c + 1)) == 0


def test_le_bruit_de_sortie_est_le_bruit_de_r_filtre() -> None:
    """Question 2.2 : `|H(j2 pi f)|^2 4 k_B R T`."""
    f, r, c, t, k = sp.symbols("f R C T k_B", positive=True)
    module = 1 / (1 + c**2 * r**2 * 4 * sp.pi**2 * f**2)
    obtenue = module * densite_spectrale_unilaterale(r, t, k)
    attendue = 4 * k * r * t / (1 + 4 * sp.pi**2 * f**2 * r**2 * c**2)
    assert sp.simplify(obtenue - attendue) == 0


def test_la_tension_mesuree_est_nulle_en_continu_et_totale_a_l_infini() -> None:
    """Question 3.2 : les deux limites, calculees et non devinees."""
    s, r, bobine, c, omega = sp.symbols("s R L C omega", positive=True)
    transfert = (s**2 * bobine * c + s * c * r) / (s**2 * bobine * c + s * c * r + 1)
    module = sp.Abs(transfert.subs(s, sp.I * omega))
    assert sp.limit(module, omega, 0) == 0
    assert sp.limit(module, omega, sp.oo) == 1


def test_le_transfert_general_est_celui_du_corrige() -> None:
    """Question 3.3 : `(R + sL)` sur `(R + sL + 1/sC)`."""
    s, r, bobine, c = sp.symbols("s R L C", positive=True)
    divise = (r + s * bobine) / (r + s * bobine + 1 / (s * c))
    attendu = (s**2 * bobine * c + s * c * r) / (s**2 * bobine * c + s * c * r + 1)
    assert sp.simplify(divise - attendu) == 0


# --- Question 4 : la chaine la moins chere -------------------------------


def test_un_seul_etage_d_opamp2_ne_suffit_pas() -> None:
    """Question 4.1 : 1 MHz sur 1 kHz plafonne a mille."""
    assert gain_maximal_par_etage(1e6, 1e3) == pytest.approx(1000.0)
    assert gain_maximal_par_etage(1e6, 1e3) < 1e4


def test_deux_etages_d_opamp2_portent_le_gain_et_coutent_moins_cher() -> None:
    """Question 4.1 : 10^6 disponible pour 4 euros, contre 10 avec OPAMP1."""
    par_etage = gain_maximal_par_etage(1e6, 1e3)
    assert par_etage**2 == pytest.approx(1e6)
    assert par_etage**2 > 1e4
    assert 2 * 2 < 10


def test_les_deux_gains_choisis_font_exactement_dix_mille() -> None:
    """Question 4.1 : `1 + 9/1` puis `1 + 999/1`."""
    premier = gain_non_inverseur(1 * KILOHM, 9 * KILOHM)
    second = gain_non_inverseur(1 * KILOHM, 999 * KILOHM)
    assert premier == pytest.approx(10.0)
    assert second == pytest.approx(1000.0)
    assert premier * second == pytest.approx(1e4)


def test_l_impedance_d_entree_du_premier_etage_ecrase_celle_de_la_source() -> None:
    """Question 4.1 : `Z_in A_ol / A_v` vaut dix gigaohms contre cinq mega."""
    impedance = 1 * MEGOHM * (1e5 / 10)
    assert impedance == pytest.approx(10e9)
    assert impedance > 5 * MEGOHM * 1e3


# --- Question 6 : le passe-haut de Butterworth ---------------------------


def test_le_passe_haut_d_ordre_trois_donne_88_pf_et_358_nh() -> None:
    """Question 6.1 : prototype `(1, 2, 1)`, 90 ohms, 20 MHz."""
    prototype = PrototypeButterworth(elements=(1.0, 2.0, 1.0), impedance=90.0)
    c1, l2, c3 = prototype.en_passe_haut(2 * math.pi * 20e6)
    assert c1 == pytest.approx(88.41 * PICOFARAD, rel=1e-3, abs=0)
    assert l2 == pytest.approx(358.1 * NANOHENRY, rel=1e-3)
    assert c3 == pytest.approx(c1, rel=1e-9, abs=0)


def test_les_358_millihenrys_du_corrige_sont_un_million_de_fois_trop() -> None:
    """Question 6.1 : le `m` ecrit vaut `n`, comme ailleurs dans ce corpus."""
    prototype = PrototypeButterworth(elements=(1.0, 2.0, 1.0), impedance=90.0)
    _, l2, _ = prototype.en_passe_haut(2 * math.pi * 20e6)
    assert l2 / 358e-3 == pytest.approx(1e-6, rel=1e-3)


def test_passer_a_quarante_cinq_ohms_divise_l_inductance_et_double_la_capacite() -> (
    None
):
    """Question 6.2 : le meme prototype, mis a l'echelle de moitie."""
    a_quarante_cinq = PrototypeButterworth(
        elements=(1.0, 2.0, 1.0),
        impedance=45.0,
    ).en_passe_haut(2 * math.pi * 20e6)
    assert a_quarante_cinq[1] == pytest.approx(179.0 * NANOHENRY, rel=1e-3)
    assert a_quarante_cinq[0] == pytest.approx(176.8 * PICOFARAD, rel=1e-3, abs=0)


# --- Question 7 : trois quadripoles bruites ------------------------------


def test_le_facteur_de_bruit_global_vaut_1_501() -> None:
    """Question 7.1 : le troisieme etage ne pese rien derriere 10^6."""
    assert facteur_d_une_cascade(
        ((1.5, 100.0), (11.0, 10.0), (5.0, 1.0)),
    ) == pytest.approx(1.501004)


def test_le_troisieme_etage_ne_change_rien_quelle_que_soit_sa_valeur() -> None:
    """Question 7.1 : c'est ce qui rend la valeur de `A_3` sans importance."""
    au_pire = facteur_d_une_cascade(((1.5, 100.0), (11.0, 10.0), (5.0, 1.0)))
    au_mieux = facteur_d_une_cascade(((1.5, 100.0), (11.0, 10.0), (1.0, 1.0)))
    assert au_pire - au_mieux < 1e-5
    assert au_mieux == pytest.approx(1.501)


# --- Question 9 : la chaine a huit microphones ---------------------------


def test_le_gain_global_de_la_chaine_vaut_5000() -> None:
    """Question 9.1 : porter 1 mV a 5 V."""
    assert gain_de_chaine(5.0, 1e-3) == pytest.approx(5000.0)


def test_chaque_signal_demande_quarante_kilohertz() -> None:
    """Question 9.1 : Nyquist sur une bande de 20 kHz."""
    assert nyquist(20e3) == pytest.approx(40e3)


def test_onze_bits_suffisent_donc_seuls_les_douze_bits_conviennent() -> None:
    """Question 9.2 : 2 mV a repartir par pas d'un microvolt."""
    assert bits_necessaires(2e-3, 1 * MICROVOLT) == 11


def test_la_cadence_agregee_vaut_320_et_non_360_kilo_echantillons() -> None:
    """Question 9.2 : huit voies a 40 kS/s.

    Le corrige ecrit 360. Sa conclusion tient -- 200 kS/s restent en deca de
    320 comme de 360 -- mais le produit est faux.
    """
    agregee = cadence_du_multiplexeur(nyquist(20e3), voies=8)
    assert agregee == pytest.approx(320e3)
    assert agregee != pytest.approx(360e3)


def test_deux_adc2_ne_tiennent_pas_la_cadence_mais_un_adc3_oui() -> None:
    """Question 9.2 : le stock limite a deux ADC2, et deux ne suffisent pas."""
    agregee = cadence_du_multiplexeur(nyquist(20e3), voies=8)
    assert agregee == pytest.approx(chaine_1_2.cadence_totale())
    deux_adc2 = chaine_1_2.cadence_disponible("ADC2")
    assert deux_adc2 == pytest.approx(200e3)
    assert deux_adc2 < agregee
    assert deux_adc2 < 360e3
    assert chaine_1_2.cadence_disponible("ADC3") == pytest.approx(400e3)
    assert agregee < chaine_1_2.cadence_disponible("ADC3")
    assert chaine_1_2.convertisseurs_possibles() == ("ADC3",)


def test_le_pire_multiplexeur_reste_sous_le_budget_de_bruit() -> None:
    """Question 9.2 : M2 a 20 microvolts par racine de hertz, divise par 5000^2."""
    rapporte = (20 * MICROVOLT) ** 2 * 20e3 / 5000.0**2
    assert rapporte == pytest.approx(0.32 * MICROVOLT**2, rel=1e-6, abs=0)
    assert rapporte < (2 * MICROVOLT) ** 2


def test_un_filtre_ou_un_multiplexeur_en_tete_ruinerait_le_budget() -> None:
    """Question 9.3 : 10 microvolts par racine de hertz sur 20 kHz."""
    depose = (10 * MICROVOLT) ** 2 * 20e3
    assert depose == pytest.approx(2e6 * MICROVOLT**2, rel=1e-9)
    assert depose > (2 * MICROVOLT) ** 2


def test_le_bruit_en_courant_est_negligeable_pour_tous_les_amplis() -> None:
    """Question 9.4 : 100 fA par racine de hertz dans 32 ohms, sur 20 kHz."""
    puissance = puissance_de_bruit_en_courant(100e-15, 32.0, 20e3)
    assert puissance == pytest.approx(2.048e-7 * MICROVOLT**2, rel=1e-3, abs=0)
    assert puissance < (2 * MICROVOLT) ** 2


def test_seul_amp3_laisse_du_budget() -> None:
    """Question 9.5 : AMP1 et AMP2 consomment tout, AMP3 un seizieme."""
    assert pytest.approx((2 * MICROVOLT) ** 2, rel=1e-9, abs=0) == (2 * MICROVOLT) ** 2
    assert (5 * MICROVOLT) ** 2 > (2 * MICROVOLT) ** 2
    assert pytest.approx(0.25 * MICROVOLT**2, rel=1e-9, abs=0) == (500e-9) ** 2


def test_le_premier_etage_peut_monter_a_1250() -> None:
    """Question 9.5 : 25 MHz sur 20 kHz."""
    assert gain_maximal_par_etage(25e6, 20e3) == pytest.approx(1250.0)


def test_le_second_etage_et_le_filtre_tiennent_dans_le_budget_restant() -> None:
    """Question 9.6 : ce que le gain de 1250 divise devient negligeable."""
    restant = (2 * MICROVOLT) ** 2 - (500e-9) ** 2
    assert restant == pytest.approx(3.75 * MICROVOLT**2, rel=1e-9, abs=0)
    ampli = (5 * MICROVOLT) ** 2 / 1250.0**2
    filtre = (10 * MICROVOLT) ** 2 * 20e3 / 1250.0**2
    assert ampli == pytest.approx(1.6e-5 * MICROVOLT**2, rel=1e-6, abs=0)
    assert filtre == pytest.approx(1.28 * MICROVOLT**2, rel=1e-6, abs=0)
    assert ampli < restant
    assert filtre < restant
