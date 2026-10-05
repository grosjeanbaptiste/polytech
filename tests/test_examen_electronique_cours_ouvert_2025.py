"""Les reponses du corrige a cours ouvert, recalculees.

Deux valeurs du manuscrit se lisent mal : l'ecriture ne separe pas « n » de
« m ». Ces tests tranchent par le calcul -- c'est la seule facon honnete de
transcrire ce qu'on ne lit pas avec certitude.
"""

import math

import pytest
import sympy as sp

from polytech.electronique.domain.acquisition import (
    bits_necessaires,
    cadence_du_multiplexeur,
    densite_de_bruit_maximale,
    gain_de_chaine,
    nyquist,
    puissance_de_bruit_en_courant,
)
from polytech.electronique.domain.amplification import gain_maximal_par_etage
from polytech.electronique.domain.filtre import PrototypeButterworth
from polytech.electronique.domain.quadripole import cellule_en_t

PICO, NANO, MICRO = 1e-12, 1e-9, 1e-6


def test_le_filtre_passe_haut_a_50_mhz_donne_63_66_pf() -> None:
    """Question 2.2 : Butterworth ordre 3, 50 ohms, coupure 50 MHz."""
    proto = PrototypeButterworth(elements=(1.0, 2.0, 1.0), impedance=50.0)
    c1, _, c3 = proto.en_passe_haut(2 * math.pi * 50e6)
    assert c1 == pytest.approx(63.66 * PICO, rel=1e-3, abs=0)
    assert c3 == pytest.approx(63.66 * PICO, rel=1e-3, abs=0)


def test_l_inductance_du_filtre_se_compte_en_nanohenrys() -> None:
    """Le manuscrit se lit « 79,6 mH » ; le calcul dit nanohenrys.

    Un filtre a 50 MHz dont les capacites valent 63,66 pF ne peut pas porter
    des millihenrys : l'ecart est d'un facteur un million. La transcription
    retient donc « nH », et ce test est ce qui l'autorise.
    """
    proto = PrototypeButterworth(elements=(1.0, 2.0, 1.0), impedance=50.0)
    _, l2, _ = proto.en_passe_haut(2 * math.pi * 50e6)
    assert l2 == pytest.approx(79.6 * NANO, rel=1e-2)
    assert l2 < 1e-6


def test_deux_cellules_rc_en_cascade_donnent_le_denominateur_du_corrige() -> None:
    """Question 5.3 : la mise en cascade n'est pas le carre du transfert."""
    s, r, c = sp.symbols("s R C", positive=True)
    chaine = cellule_en_t(r, sp.Integer(0), 1 / (s * c)) ** 2
    transfert = 1 / chaine[0, 0]
    attendu = 1 / (s**2 * r**2 * c**2 + 3 * s * r * c + 1)
    assert sp.simplify(transfert - attendu) == 0


def test_la_cascade_n_est_pas_le_carre_du_transfert_isole() -> None:
    """C'est tout le propos de la question : la seconde cellule charge la premiere."""
    s, r, c = sp.symbols("s R C", positive=True)
    naif = 1 / (s * r * c + 1) ** 2
    reel = 1 / (cellule_en_t(r, sp.Integer(0), 1 / (s * c)) ** 2)[0, 0]
    assert sp.simplify(reel - naif) != 0


def test_le_gain_de_chaine_vaut_125000() -> None:
    """Question 7.1 : 5 V en sortie pour 40 microvolts en entree."""
    assert gain_de_chaine(5.0, 40 * MICRO) == pytest.approx(125_000)


def test_la_cadence_minimale_est_de_400_hertz() -> None:
    assert nyquist(200.0) == pytest.approx(400.0)


def test_le_multiplexeur_commute_a_3200_hertz() -> None:
    """Question 7.2 : huit capteurs echantillonnes chacun a 400 Hz."""
    assert cadence_du_multiplexeur(400.0, 8) == pytest.approx(3200.0)


def test_onze_bits_sont_necessaires_pour_le_pas_demande() -> None:
    """Question 7.3 : 80 microvolts d'etendue, pas de 0,05 microvolt."""
    assert bits_necessaires(80 * MICRO, 0.05 * MICRO) == 11


def test_le_cahier_des_charges_est_effectivement_impossible() -> None:
    """Aucun ADC ne satisfait a la fois la resolution et la cadence.

    Le correcteur l'a note « question impossible » ; le calcul le confirme,
    et c'est ce qui vaut au corrige son point bonus.
    """
    adcs = {"ADC1": (8, 40e3), "ADC2": (10, 15e3), "ADC3": (14, 2e3)}
    bits = bits_necessaires(80 * MICRO, 0.05 * MICRO)
    cadence = cadence_du_multiplexeur(nyquist(200.0), 8)
    assert [n for n, (b, _) in adcs.items() if b >= bits] == ["ADC3"]
    assert [n for n, (_, f) in adcs.items() if f >= cadence] == ["ADC1", "ADC2"]
    assert not [n for n, (b, f) in adcs.items() if b >= bits and f >= cadence]


def test_le_bruit_en_courant_est_negligeable() -> None:
    """Question 7.4 : 20 fA/racine(Hz) dans 1 kilohm sur 200 Hz."""
    puissance = puissance_de_bruit_en_courant(20e-15, 1e3, 200.0)
    assert puissance == pytest.approx(8e-20, rel=1e-2, abs=0)
    assert puissance < (0.75 * MICRO) ** 2 / 1000


def test_seul_amp3_tient_le_budget_de_bruit() -> None:
    """Question 7.5 : la densite maximale vaut 53 nV/racine(Hz), non 53 mV.

    C'est la seconde lecture incertaine du manuscrit, et elle se tranche
    seule : a 53 mV, les trois amplificateurs passeraient et la conclusion
    « AMP3 uniquement » serait fausse.
    """
    densite = densite_de_bruit_maximale(0.75 * MICRO, 200.0)
    assert densite == pytest.approx(53 * NANO, rel=1e-2)
    amplis = {"AMP1": 70 * NANO, "AMP2": 60 * NANO, "AMP3": 15 * NANO}
    assert [n for n, d in amplis.items() if d <= densite] == ["AMP3"]


def test_le_gain_maximal_d_amp3_vaut_dix_mille() -> None:
    assert gain_maximal_par_etage(2e6, 200.0) == pytest.approx(1e4)
