"""Les reponses du corrige manuscrit, recalculees au lieu d'etre crues.

La transcription de
`ElectroniqueInstrumentation/EXAMENS/ElecInstruTheorie202425Corrige_annote.pdf`
porte des resultats chiffres. Les lire ne prouve rien : ces tests les
refont depuis les donnees de l'enonce, et echouent si la transcription dit
autre chose que ce que la physique donne.
"""

import pytest
import sympy as sp

from polytech.electronique.domain.amplification import (
    gain_inverseur,
    gain_maximal_par_etage,
    gain_non_inverseur,
    nombre_d_etages,
)
from polytech.electronique.domain.bruit import densite_spectrale_unilaterale
from polytech.electronique.domain.conversion import ConvertisseurAnalogiqueNumerique
from polytech.electronique.domain.transfert import DiviseurRL

MILLIVOLT = 1e-3


def test_le_pas_de_quantification_vaut_bien_3_66_millivolts() -> None:
    """Question 1.3 : ADC 12 bits alimente en +15 V / 0 V."""
    adc = ConvertisseurAnalogiqueNumerique(bits=12, pleine_echelle_volts=15.0)
    assert adc.pas_approche() == pytest.approx(3.66 * MILLIVOLT, rel=1e-3)


def test_l_approximation_du_corrige_est_legitime() -> None:
    """Il ecrit `2^n - 1` puis approxime par `2^n` : l'ecart se chiffre.

    A douze bits, l'ecart relatif vaut 1/4095, soit moins d'un millieme --
    trois fois moins que le dernier chiffre affiche. L'approximation ne
    change donc pas la reponse, et c'est cela qui la rend legitime.
    """
    adc = ConvertisseurAnalogiqueNumerique(bits=12, pleine_echelle_volts=15.0)
    ecart = abs(adc.pas_exact() - adc.pas_approche()) / adc.pas_exact()
    assert ecart < 1e-3


def test_la_densite_de_bruit_est_quatre_k_b_r_t() -> None:
    """Question 1.4 : PSD unilaterale du bruit thermique d'une resistance."""
    resistance, temperature, constante = sp.symbols("R T k_B", positive=True)
    obtenue = densite_spectrale_unilaterale(resistance, temperature, constante)
    assert sp.simplify(obtenue - 4 * constante * resistance * temperature) == 0


def test_la_fonction_de_transfert_est_celle_du_corrige() -> None:
    """Question 2.1 : diviseur de tension sur la branche R_L + sL."""
    s = sp.Symbol("s")
    r, r_l, inductance = sp.symbols("R R_L L", positive=True)
    attendue = (s + r_l / inductance) / (s + (r + r_l) / inductance)
    obtenue = DiviseurRL(r, r_l, inductance).fonction_de_transfert()
    assert sp.simplify(obtenue - attendue) == 0


def test_le_pole_et_le_zero_valent_ce_que_le_corrige_annonce() -> None:
    """Question 2.2 : R_L = 1 ohm, R = 100 ohms, L = 100 microhenrys."""
    circuit = DiviseurRL(100.0, 1.0, 100e-6)
    assert float(circuit.zero()) == pytest.approx(-1e4)
    assert float(circuit.pole()) == pytest.approx(-1.01e6, rel=1e-2)


def test_les_deux_gains_asymptotiques_valent_moins_40_db_et_0_db() -> None:
    circuit = DiviseurRL(100.0, 1.0, 100e-6)
    assert circuit.gain_continu_en_decibels() == pytest.approx(-40.0, abs=0.1)
    assert circuit.gain_infini_en_decibels() == pytest.approx(0.0, abs=1e-9)


def test_un_seul_etage_ne_suffit_pas_et_deux_suffisent() -> None:
    """Question 3.1 : GBW 2,5 MHz, bande passante 100 kHz, gain vise 500."""
    par_etage = gain_maximal_par_etage(2.5e6, 100e3)
    assert par_etage == pytest.approx(25.0)
    assert par_etage < 500
    assert par_etage**2 == pytest.approx(625.0)
    assert nombre_d_etages(500.0, 2.5e6, 100e3) == 2


def test_les_resistances_choisies_donnent_les_gains_annonces() -> None:
    """Question 3.2 : 1 k / 24 k en non-inverseur, 1 k / 20 k en inverseur."""
    assert gain_non_inverseur(1e3, 24e3) == pytest.approx(25.0)
    assert gain_inverseur(1e3, 20e3) == pytest.approx(-20.0)


def test_la_chaine_des_deux_etages_rend_bien_le_gain_vise() -> None:
    """C'est la verification que le corrige ne fait pas explicitement."""
    produit = gain_non_inverseur(1e3, 24e3) * gain_inverseur(1e3, 20e3)
    assert produit == pytest.approx(-500.0)
