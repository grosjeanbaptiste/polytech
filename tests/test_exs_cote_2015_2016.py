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

import math

import sympy as sp

from polytech.electromagnetisme.application import cote_2016_couples as co
from polytech.electromagnetisme.application import cote_2016_enonce as en
from polytech.electromagnetisme.application import cote_2016_magnetique as ma
from polytech.electromagnetisme.application import cote_2016_regime as re
from polytech.electromagnetisme.application import cote_2016_sinusoidal as si
from polytech.maths.domain.comparaison import (
    argument_en_degres,
    egales,
    proches,
    proches_relativement,
)

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
    assert proches_relativement(float(energie), 3.67e-7, 2e-3)
    assert energie == sp.Rational(18, 49) * sp.Rational(1, 10**6)


# --- Question 2 : le circuit magnetique a deux fenetres ---------------------


def test_q2_les_flux_et_les_inductions() -> None:
    """Page 3 : `Phi_c = 4e-4` Wb, la moitie dans chaque colonne, 1 T partout."""
    etat = ma.etat_des_colonnes()
    assert proches_relativement(etat.flux_central, 4e-4, 1e-12)
    assert proches_relativement(etat.flux_lateral, 2e-4, 1e-12)
    assert proches_relativement(etat.induction_laterale, 1.0, 1e-12)


def test_q2_les_longueurs_des_contours() -> None:
    """Page 3 : 3,5 cm pour la colonne centrale, 9,5 cm pour le reste du tour."""
    centrale, laterale = ma.longueurs_des_contours()
    assert proches_relativement(centrale, 0.035, 1e-12)
    assert proches_relativement(laterale, 0.095, 1e-12)


def test_q2_le_courant() -> None:
    """Page 3 : `I = 0,325` A."""
    assert proches_relativement(ma.courant_des_bobines(), 0.325, 1e-12)


def test_q2_l_inductance_donnee_ne_decrit_pas_cet_etat() -> None:
    """`N Phi_1 / I` vaut 61,5 mH : loin du henry que l'enonce donne a `L11`."""
    apparente = ma.inductance_apparente()
    assert proches_relativement(apparente, 100 * 2e-4 / 0.325, 1e-12)
    assert apparente < en.INDUCTANCE_PROPRE / 10
    assert proches_relativement(apparente, 61.5e-3, 1e-3)
    assert 16 < en.INDUCTANCE_PROPRE / apparente < 17


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
    assert egales(premiere, attendue_1)
    assert egales(seconde, attendue_2)


def test_q3_le_regime_par_la_valeur_finale() -> None:
    """Page 4 : `i1 = E1/R12`, `i2 = E2/R2` -- retrouves depuis les images."""
    assert co.regime() == (co.E1 / co.R12, co.E2 / co.R2)


def test_q3_les_courants_ne_sautent_pas_a_la_fermeture() -> None:
    """La valeur initiale des images redonne les conditions initiales."""
    assert co.valeurs_initiales() == co.conditions_initiales()


def test_q3_la_pente_du_courant_secondaire_a_la_fermeture() -> None:
    """`di2/dt(0+) = E2 L1 / (L1 L2 - L12^2)` : le couplage raidit le depart."""
    attendue = co.E2 * co.L1 / (co.L1 * co.L2 - co.L12**2)
    assert egales(co.pente_initiale_du_secondaire(), attendue)


# --- Question 4 : regime sinusoidal ----------------------------------------


def test_q4_les_courants_donnes_par_l_enonce_resolvent_le_systeme() -> None:
    """`I1 = 11,012 - 0,375j`, `I2 = -10,968 + 0,314j`, a trois decimales."""
    i1, i2 = si.courants_de_mailles()
    assert proches(i1, en.I1_DONNE, 1e-3)
    assert proches(i2, en.I2_DONNE, 1e-3)
    assert complex(11.012, -0.375) == en.I1_DONNE
    assert complex(-10.968, 0.314) == en.I2_DONNE


def test_q4_la_tension_de_la_branche() -> None:
    """Page 5 : `U = 109,88 + 3,75j`, de module 109,94 et d'argument 1,95 degre."""
    tension = si.tension_de_la_branche()
    assert proches(tension.real, 109.88, 5e-3)
    assert proches(tension.imag, 3.75, 5e-3)
    assert proches(abs(tension), 109.94, 5e-3)
    assert proches(argument_en_degres(tension), 1.95, 6e-3)


def test_q4_la_puissance_complexe() -> None:
    """Page 5 : `S = 1204,0 + 75,6j`."""
    puissance = si.puissance_de_la_branche()
    assert proches(puissance.real, 1204.0, 0.05)
    assert proches(puissance.imag, 75.6, 0.05)


def test_q4_la_puissance_par_le_module_du_courant() -> None:
    """Second chemin : `P = R3 |I|^2`, `Q = omega L3 |I|^2`."""
    active, reactive = si.puissances_par_le_courant()
    puissance = si.puissance_de_la_branche()
    assert proches_relativement(active, puissance.real, 1e-12)
    assert proches_relativement(reactive, puissance.imag, 1e-12)


def test_q4_le_dephasage_du_courant_sur_la_tension() -> None:
    """Page 5 : -3,59 degres (tronque) ; c'est `-arctan(omega L3 / R3)`."""
    dephasage = si.dephasage_du_courant()
    assert proches(dephasage, -3.59, 6e-3)
    par_l_impedance = -argument_en_degres(complex(en.R3, en.PULSATION * en.L3))
    assert proches_relativement(dephasage, par_l_impedance, 1e-9)


def test_q4_l_argument_de_la_tension_est_positif() -> None:
    """Le manuscrit ecrit `-(-1,95)` a sa derniere ligne ; l'argument vaut +1,95."""
    tension, courant = si.tension_de_la_branche(), si.courant_de_la_branche()
    assert argument_en_degres(tension) > 0
    assert proches(argument_en_degres(courant), -1.64, 5e-3)


# --- Question 5 : ouverture en regime sinusoidal ---------------------------


def test_q5_le_condensateur_est_decharge_a_l_ouverture() -> None:
    """Page 6 : `e(0) = 0`, donc `u_C(0) = 0` et `i(t) = 0`."""
    assert proches(si.tension_a_l_ouverture(), 0.0, 1e-15)
    assert proches(si.courant_apres_ouverture(1e-5), 0.0, 1e-15)


def test_q5_ouvert_a_un_autre_instant_le_courant_s_eteindrait() -> None:
    """Avec une phase nulle : `E0/R` au depart, puis `e^(-t/RC)`."""
    constante = en.R * en.C
    depart = si.courant_apres_ouverture(0.0, phase=0.0)
    assert proches_relativement(depart, en.E0 / en.R, 1e-12)
    plus_tard = si.courant_apres_ouverture(constante, phase=0.0)
    assert proches_relativement(plus_tard, depart / math.e, 1e-12)
    assert proches_relativement(constante, 1e-5, 1e-12)


# --- Question 6 : deux demi-tores -------------------------------------------


def test_q6_le_nombre_de_spires() -> None:
    """Page 7 : `Phi = 4,744e-4` Wb, `N = 2000`."""
    assert proches_relativement(ma.flux_du_tore(), 4.744e-4, 2e-4)
    assert ma.spires_du_tore() == 2000


def test_q6_l_ecartement_des_demi_tores() -> None:
    """Page 7 : `H_e = 1 201 619,82` A/m, `l_e = 0,8257` mm."""
    assert proches_relativement(ma.champ_d_entrefer(), 1_201_619.82, 1e-6)
    assert proches(ma.ecartement() * 1e3, 0.8257, 5e-5)


def test_q6_la_boucle_d_ampere_se_referme() -> None:
    """Le circuit reconstruit avec cet ecartement redonne `N I = 2000` A."""
    circuit = ma.circuit_ecarte()
    assert circuit.entrefers == 2
    assert proches_relativement(circuit.amperes_tours(), 2000.0, 1e-12)
    assert circuit.part_des_entrefers() > 0.99


def test_q6_sans_entrefer_le_fer_sature() -> None:
    """Page 7 : `H = 6366,198` A/m, donc 1,8 T, et `L = 1,131` H."""
    champ, inductance = ma.demi_tores_jointifs()
    assert proches_relativement(champ, 6366.198, 1e-6)
    assert champ > en.CHAMP_DE_SATURATION
    assert proches(inductance, 1.131, 5e-4)


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
    assert proches(i1, -i2, abs(i1) / 100)
    assert 24_000 < ma.champ_d_entrefer() / en.CHAMP_VOULU < 24_100
    champ, _ = ma.demi_tores_jointifs()
    assert 25 < champ / en.CHAMP_DE_SATURATION < 26
    periode = 1 / en.FREQUENCE
    assert proches_relativement(periode / (en.R * en.C), 2000, 1e-12)
    assert 1 / (1 - ma.circuit_ecarte().part_des_entrefers()) > 100
    assert 2 * ma.ecartement() < 2e-3
