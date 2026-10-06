"""Electromagnetisme : l'exercice cote du 19 mai 2016, six questions.

Sources :

- l'enonce, `Electromagnetisme/EXS/Exs_Cote_2015_2016_Enonce.pdf` ;
- la solution manuscrite, `Electromagnetisme/EXS/Exs_Cote_2015_2016_Solution.pdf`,
  sept pages, transcrite dans
  `transcriptions/Electromagnetisme/EXS/Exs_Cote_2015_2016_Solution.md`.

Chaque nombre du manuscrit est recopie comme oracle, et recoupe par un
chemin qu'il n'emprunte pas : Millman pour Thevenin, `R |I|^2` pour la
puissance, l'arc tangente de `omega L / R` pour le dephasage, la boucle
fermee d'Ampere pour l'entrefer.
"""

from __future__ import annotations

import cmath
import math

import pytest
import sympy as sp

from polytech.electromagnetisme.application import cote_2016_couples as co
from polytech.electromagnetisme.application import cote_2016_enonce as en
from polytech.electromagnetisme.application import cote_2016_magnetique as ma
from polytech.electromagnetisme.application import cote_2016_regime as re
from polytech.electromagnetisme.application import cote_2016_sinusoidal as si

# --- Question 1 : Thevenin, puis l'energie du condensateur -----------------


def test_q1_le_generateur_equivalent_du_manuscrit() -> None:
    """Page 1 et 2 : `R0 = 1/3` ohm, `E0 = 1` V."""
    tension, resistance = re.thevenin()
    assert tension == 1
    assert resistance == sp.Rational(1, 3)


def test_q1_millman_donne_la_meme_tension() -> None:
    """Second chemin : trois branches en parallele entre A et B."""
    assert re.tension_par_millman() == re.thevenin()[0]


def test_q1_le_court_circuit_recoupe_la_resistance() -> None:
    """`I_cc = E0 / R0` : trois amperes, mesures sur le reseau lui-meme."""
    tension, resistance = re.thevenin()
    assert re.courant_de_court_circuit() == tension / resistance == 3


def test_q1_la_branche_du_condensateur_ne_compte_pas() -> None:
    """En regime `C2` est ouvert : `E2` peut valoir n'importe quoi."""
    assert re.thevenin(e2=sp.Integer(50)) == re.thevenin()


def test_q1_l_energie_emmagasinee() -> None:
    """Page 2 : `I = 3/7` A, `U = 6/7` V, `3,67e-7` J."""
    courant, tension, energie = re.charge_du_dipole()
    assert courant == sp.Rational(3, 7)
    assert tension == sp.Rational(6, 7)
    assert float(energie) == pytest.approx(3.67e-7, rel=2e-3, abs=0)
    assert energie == sp.Rational(18, 49) * sp.Rational(1, 10**6)


# --- Question 2 : le circuit magnetique a deux fenetres ---------------------


def test_q2_les_flux_et_les_inductions() -> None:
    """Page 3 : `Phi_c = 4e-4` Wb, la moitie dans chaque colonne, 1 T partout."""
    etat = ma.etat_des_colonnes()
    assert etat.flux_central == pytest.approx(4e-4, rel=1e-12, abs=0)
    assert etat.flux_lateral == pytest.approx(2e-4, rel=1e-12, abs=0)
    assert etat.induction_laterale == pytest.approx(1.0, rel=1e-12, abs=0)


def test_q2_les_longueurs_des_contours() -> None:
    """Page 3 : 3,5 cm pour la colonne centrale, 9,5 cm pour le reste du tour."""
    centrale, laterale = ma.longueurs_des_contours()
    assert centrale == pytest.approx(0.035, rel=1e-12, abs=0)
    assert laterale == pytest.approx(0.095, rel=1e-12, abs=0)


def test_q2_le_courant() -> None:
    """Page 3 : `I = 0,325` A."""
    assert ma.courant_des_bobines() == pytest.approx(0.325, rel=1e-12, abs=0)


def test_q2_l_inductance_donnee_ne_decrit_pas_cet_etat() -> None:
    """`N Phi_1 / I` vaut 61,5 mH : loin du henry que l'enonce donne a `L11`."""
    apparente = ma.inductance_apparente()
    assert apparente == pytest.approx(100 * 2e-4 / 0.325, rel=1e-12, abs=0)
    assert apparente < en.INDUCTANCE_PROPRE / 10


# --- Question 3 : deux circuits couples ------------------------------------


def test_q3_les_conditions_initiales() -> None:
    """Page 4 : `i1(0) = E1/R12`, `i2(0) = 0`."""
    assert co.conditions_initiales() == (co.E1 / co.R12, 0)


def test_q3_le_systeme_en_laplace_du_manuscrit() -> None:
    """Page 4, derniere accolade : les deux equations en `I1(p)`, `I2(p)`."""
    p, i1, i2 = co.P, co.I1, co.I2
    premiere, seconde = co.systeme_en_laplace()
    attendue_1 = (
        (co.R12 + p * co.L1) * i1 + p * co.L12 * i2 - co.L1 / co.R12 * co.E1 - co.E1 / p
    )
    attendue_2 = (
        p * co.L12 * i1 + (co.R2 + p * co.L2) * i2 - co.L12 / co.R12 * co.E1 - co.E2 / p
    )
    assert sp.simplify(premiere - attendue_1) == 0
    assert sp.simplify(seconde - attendue_2) == 0


def test_q3_le_regime_par_la_valeur_finale() -> None:
    """Page 4 : `i1 = E1/R12`, `i2 = E2/R2` -- retrouves depuis les images."""
    assert co.regime() == (co.E1 / co.R12, co.E2 / co.R2)


def test_q3_les_courants_ne_sautent_pas_a_la_fermeture() -> None:
    """La valeur initiale des images redonne les conditions initiales."""
    assert co.valeurs_initiales() == co.conditions_initiales()


def test_q3_la_pente_du_courant_secondaire_a_la_fermeture() -> None:
    """`di2/dt(0+) = E2 L1 / (L1 L2 - L12^2)` : le couplage raidit le depart."""
    attendue = co.E2 * co.L1 / (co.L1 * co.L2 - co.L12**2)
    assert sp.simplify(co.pente_initiale_du_secondaire() - attendue) == 0


# --- Question 4 : regime sinusoidal ----------------------------------------


def test_q4_les_courants_donnes_par_l_enonce_resolvent_le_systeme() -> None:
    """`I1 = 11,012 - 0,375j`, `I2 = -10,968 + 0,314j`, a trois decimales."""
    i1, i2 = si.courants_de_mailles()
    assert abs(i1 - en.I1_DONNE) < 1e-3
    assert abs(i2 - en.I2_DONNE) < 1e-3
    assert complex(11.012, -0.375) == en.I1_DONNE
    assert complex(-10.968, 0.314) == en.I2_DONNE


def test_q4_la_tension_de_la_branche() -> None:
    """Page 5 : `U = 109,88 + 3,75j`, de module 109,94 et d'argument 1,95 degre."""
    tension = si.tension_de_la_branche()
    assert tension.real == pytest.approx(109.88, abs=5e-3)
    assert tension.imag == pytest.approx(3.75, abs=5e-3)
    assert abs(tension) == pytest.approx(109.94, abs=5e-3)
    assert math.degrees(cmath.phase(tension)) == pytest.approx(1.95, abs=6e-3)


def test_q4_la_puissance_complexe() -> None:
    """Page 5 : `S = 1204,0 + 75,6j`."""
    puissance = si.puissance_de_la_branche()
    assert puissance.real == pytest.approx(1204.0, abs=0.05)
    assert puissance.imag == pytest.approx(75.6, abs=0.05)


def test_q4_la_puissance_par_le_module_du_courant() -> None:
    """Second chemin : `P = R3 |I|^2`, `Q = omega L3 |I|^2`."""
    active, reactive = si.puissances_par_le_courant()
    puissance = si.puissance_de_la_branche()
    assert active == pytest.approx(puissance.real, rel=1e-12, abs=0)
    assert reactive == pytest.approx(puissance.imag, rel=1e-12, abs=0)


def test_q4_le_dephasage_du_courant_sur_la_tension() -> None:
    """Page 5 : -3,59 degres (tronque) ; c'est `-arctan(omega L3 / R3)`."""
    dephasage = si.dephasage_du_courant()
    assert dephasage == pytest.approx(-3.59, abs=6e-3)
    par_l_impedance = -math.degrees(math.atan(en.PULSATION * en.L3 / en.R3))
    assert dephasage == pytest.approx(par_l_impedance, rel=1e-9, abs=0)


def test_q4_l_argument_de_la_tension_est_positif() -> None:
    """Le manuscrit ecrit `-(-1,95)` a sa derniere ligne ; l'argument vaut +1,95."""
    tension, courant = si.tension_de_la_branche(), si.courant_de_la_branche()
    assert math.degrees(cmath.phase(tension)) > 0
    assert math.degrees(cmath.phase(courant)) == pytest.approx(-1.64, abs=5e-3)


# --- Question 5 : ouverture en regime sinusoidal ---------------------------


def test_q5_le_condensateur_est_decharge_a_l_ouverture() -> None:
    """Page 6 : `e(0) = 0`, donc `u_C(0) = 0` et `i(t) = 0`."""
    assert si.tension_a_l_ouverture() == pytest.approx(0.0, abs=1e-15)
    assert si.courant_apres_ouverture(1e-5) == pytest.approx(0.0, abs=1e-15)


def test_q5_ouvert_a_un_autre_instant_le_courant_s_eteindrait() -> None:
    """Avec une phase nulle : `E0/R` au depart, puis `e^(-t/RC)`."""
    constante = en.R * en.C
    depart = si.courant_apres_ouverture(0.0, phase=0.0)
    assert depart == pytest.approx(en.E0 / en.R, rel=1e-12, abs=0)
    plus_tard = si.courant_apres_ouverture(constante, phase=0.0)
    assert plus_tard == pytest.approx(depart / math.e, rel=1e-12, abs=0)
    assert constante == pytest.approx(1e-5, rel=1e-12, abs=0)


# --- Question 6 : deux demi-tores -------------------------------------------


def test_q6_le_nombre_de_spires() -> None:
    """Page 7 : `Phi = 4,744e-4` Wb, `N = 2000`."""
    assert ma.flux_du_tore() == pytest.approx(4.744e-4, rel=2e-4, abs=0)
    assert ma.spires_du_tore() == 2000


def test_q6_l_ecartement_des_demi_tores() -> None:
    """Page 7 : `H_e = 1 201 619,82` A/m, `l_e = 0,8257` mm."""
    assert ma.champ_d_entrefer() == pytest.approx(1_201_619.82, rel=1e-6, abs=0)
    assert ma.ecartement() * 1e3 == pytest.approx(0.8257, abs=5e-5)


def test_q6_la_boucle_d_ampere_se_referme() -> None:
    """Le circuit reconstruit avec cet ecartement redonne `N I = 2000` A."""
    circuit = ma.circuit_ecarte()
    assert circuit.entrefers == 2
    assert circuit.amperes_tours() == pytest.approx(2000.0, rel=1e-12, abs=0)
    assert circuit.part_des_entrefers() > 0.99


def test_q6_sans_entrefer_le_fer_sature() -> None:
    """Page 7 : `H = 6366,198` A/m, donc 1,8 T, et `L = 1,131` H."""
    champ, inductance = ma.demi_tores_jointifs()
    assert champ == pytest.approx(6366.198, rel=1e-6, abs=0)
    assert champ > en.CHAMP_DE_SATURATION
    assert inductance == pytest.approx(1.131, abs=5e-4)


def test_q1_la_constante_de_temps_de_la_charge() -> None:
    """`C` se charge a travers `R0` en parallele avec `R` : `2/7` microseconde."""
    assert re.constante_de_charge() == sp.Rational(2, 7) * sp.Rational(1, 10**6)


# --- Les ordres de grandeur que le corrige annonce en toutes lettres -------


def test_les_ordres_de_grandeur_du_document() -> None:
    """Chaque « tant de fois » du texte est un quotient que l'on recalcule ici."""
    assert 16 < en.INDUCTANCE_PROPRE / ma.inductance_apparente() < 17
    reactance = 1 / (en.PULSATION * en.C2)
    assert 155 < reactance / en.R2_Q4 < 165
    i1, i2 = si.courants_de_mailles()
    assert abs(i1 + i2) < abs(i1) / 100
    assert 24_000 < ma.champ_d_entrefer() / en.CHAMP_VOULU < 24_100
    champ, _ = ma.demi_tores_jointifs()
    assert 25 < champ / en.CHAMP_DE_SATURATION < 26
    periode = 1 / en.FREQUENCE
    assert periode / (en.R * en.C) == pytest.approx(2000, rel=1e-12, abs=0)
    assert 1 / (1 - ma.circuit_ecarte().part_des_entrefers()) > 100
    assert 2 * ma.ecartement() < 2e-3
