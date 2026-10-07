"""Signaux et systemes : les exercices resolus, chapitre 1 (0.1 a 0.10).

Source : `SignauxSystemes/EXS/Exs_01_02_04_05_Resolus.pdf`, pages 1 a 4 --
enonces, puis « Solutions » imprimees. Chaque solution est recopiee comme
oracle et recoupee par un calcul independant. Deux d'entre elles sont
contredites : le signe de la variation du rapport signal sur bruit (0.6)
et un morceau du produit de convolution (0.10).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.signaux.application import resolus_convolutions as cv
from polytech.signaux.application import resolus_puissances as pu
from polytech.signaux.application import resolus_signaux as sg
from polytech.signaux.application import resolus_trains as tn
from polytech.signaux.domain.frequence import INSTANT

# --- 0.2 et 0.3 : un triangle comprime, deux energies -----------------------


def test_0_2_le_triangle_de_la_solution() -> None:
    """Page 3 : « un triangle entre t = 1/3 et 1, d'amplitude 1/2 »."""
    debut, sommet, fin, hauteur = sg.reperes_du_triangle()
    assert (debut, fin, hauteur) == (sp.Rational(1, 3), 1, sp.Rational(1, 2))
    assert sommet == sp.Rational(2, 3)


def test_0_3_les_energies_de_la_solution() -> None:
    """Page 3 : `E(rect) = 1`, `E(tri) = 2/3`."""
    assert sg.energies() == (1, sp.Rational(2, 3))


def test_0_3_amplifier_par_deux_quadruple_l_energie() -> None:
    """Page 3 : « l'energie est multipliee par 4 »."""
    assert sg.energie_amplifiee(2) == (4, sp.Rational(8, 3))


# --- 0.4 et 0.5 : des valeurs efficaces -------------------------------------


def test_0_4_les_valeurs_efficaces_de_la_solution() -> None:
    """Page 3 : `racine(2)/2`, `racine(2)` et 1."""
    assert pu.valeurs_efficaces_des_trains() == (sp.sqrt(2) / 2, sp.sqrt(2), 1)


def test_0_4_le_train_de_triangles_est_constant() -> None:
    """Second chemin : des triangles de base 2 repetes chaque seconde font 1."""
    for instant in (0, sp.Rational(1, 4), sp.Rational(7, 10)):
        assert pu.train_de_triangles(instant) == 1


def test_0_4_un_motif_qui_ne_couvre_pas_l_instant_est_refuse() -> None:
    """Un motif sans branche par defaut ne dit rien hors de son domaine."""
    troue = sp.Piecewise((1, INSTANT < 0))
    with pytest.raises(ValueError, match="aucun morceau"):
        pu.train(troue, sp.Integer(10), sp.Integer(1))


def test_0_5_les_valeurs_efficaces_de_la_solution() -> None:
    """Page 3 : `a/racine 2`, `a`, `a/racine 3`, `a/racine 3`."""
    a = pu.A
    assert pu.valeurs_efficaces_des_formes() == {
        "sinus": a / sp.sqrt(2),
        "rectangle": a,
        "triangle": a / sp.sqrt(3),
        "rampe": a / sp.sqrt(3),
    }


# --- 0.6 : un rapport signal sur bruit --------------------------------------


def test_0_6_le_rapport_de_la_solution() -> None:
    """Page 3 : `20 log10(1/racine 2) = -3` dB."""
    assert pu.rapport_signal_sur_bruit() == pytest.approx(-3.01, abs=5e-3)


def test_0_6_quarante_decibels_font_un_facteur_cent() -> None:
    """Page 3 : « un facteur 100 (en amplitude) »."""
    assert pu.facteur_d_amplitude(40) == pytest.approx(100)


def test_0_6_attenuer_le_bruit_fait_monter_le_rapport() -> None:
    """La solution annonce -43 dB ; attenuer le bruit de 100 donne +37 dB."""
    assert pu.rapport_signal_sur_bruit(gain_du_bruit=0.01) == pytest.approx(
        36.99, abs=5e-3
    )
    assert pu.rapport_signal_sur_bruit(gain_du_bruit=100) == pytest.approx(
        -43.01, abs=5e-3
    )


# --- 0.7 et 0.8 : produits scalaires, energies transformees -----------------


def test_0_7_a_le_produit_scalaire_de_la_solution() -> None:
    """Page 4 : `<rect, tri> = 3/4`."""
    assert sg.produit_scalaire_des_formes() == sp.Rational(3, 4)


def test_0_7_b_la_regle_du_cours() -> None:
    """Page 4 : zero, les periodes 2 et 3/4 n'etant pas multiples (syllabus, 1.39)."""
    assert tn.produit_scalaire_des_trains() == 0
    assert not tn.sont_multiples(sp.Integer(2), sp.Rational(3, 4))


def test_0_7_b_des_periodes_multiples_sortent_de_la_regle() -> None:
    """Une periode multiple de l'autre : (1.39) demande alors l'integrale."""
    with pytest.raises(NotImplementedError, match="periodes multiples"):
        tn.produit_scalaire_des_trains(sp.Integer(2), sp.Integer(1))


def test_0_7_b_la_moyenne_du_produit_n_est_pas_nulle() -> None:
    """Sur la periode commune de 6 s, la moyenne du produit vaut 2/3."""
    assert tn.moyenne_du_produit_des_trains() == sp.Rational(2, 3)
    assert tn.produit_des_moyennes_des_trains() == sp.Rational(2, 3)


def test_0_7_b_la_periode_commune() -> None:
    """Deux secondes et trois quarts de seconde se retrouvent toutes les six."""
    assert tn.periode_commune() == 6
    assert tn.periode_commune(sp.Rational(1, 2), sp.Rational(3, 4)) == sp.Rational(3, 2)
    assert tn.sont_multiples(sp.Integer(1), sp.Integer(2))
    assert tn.sont_multiples(sp.Rational(1, 2), sp.Rational(1, 4))


def test_0_8_les_energies_de_la_solution() -> None:
    """Page 4 : `E/2` pour `x(2t)`, `E` pour `x(t - 2)`."""
    for signal in sg.SIGNAUX_D_ESSAI:
        assert sg.rapports_d_energie(signal) == (sp.Rational(1, 2), 1)


# --- 0.9 et 0.10 : deux produits de convolution -----------------------------


def test_0_9_la_valeur_de_la_solution() -> None:
    """Page 4 : `y(4) = 6`, l'aire du trapeze hachure entre 1 et 2."""
    assert cv.sortie_de_la_rampe(4) == 6
    assert cv.aire_du_trapeze_hachure() == 6


def test_0_10_les_morceaux() -> None:
    """Zero jusqu'a 1, puis `t - 1`, puis 2 : l'integrale courante de la porte."""
    assert cv.sortie_de_l_echelon(1) == 0
    assert cv.sortie_de_l_echelon(2) == 1
    assert cv.sortie_de_l_echelon(3) == 2
    assert cv.sortie_de_l_echelon(7) == 2


def test_0_10_le_morceau_du_milieu_n_est_pas_t() -> None:
    """Page 4 : « x(t) = t pour 1 < t <= 3 » ; c'est `t - 1`, seul continu."""
    milieu = sp.Rational(5, 2)
    assert cv.sortie_de_l_echelon(milieu) == milieu - 1
    assert cv.sortie_de_l_echelon(milieu) != milieu
    assert cv.saut_si_le_milieu_valait_t() == (1, -1)
