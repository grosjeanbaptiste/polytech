"""Signaux et systemes : le recueil d'exercices, planches sur les signaux.

Source : `SignauxSystemes/EXS/Exs_Recueil.pdf`, dix-neuf planches sans
numero d'exercice ; chacune est designee ici par son numero de planche.
Ce fichier couvre les planches 1, 2, 3, 13, 17 et 18 : convolutions et
spectres. Les reponses que les planches portent -- imprimees ou
manuscrites -- sont recopiees comme oracles, et recoupees par un calcul
independant.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.signaux.application import recueil_appariement as ap
from polytech.signaux.application import recueil_convolutions as cv
from polytech.signaux.application import recueil_spectres as sx

# --- Planche 1 : quelle entree, quelle reponse impulsionnelle ? -------------


def test_p1_l_appariement_de_la_planche() -> None:
    """Planche 1 : `y1 = x1*h1`, `y2 = x2*h1`, `y3 = x1*h2`, `y4 = x2*h2`."""
    assert ap.appariement() == {
        "y1": ("x1", "h1"),
        "y2": ("x2", "h1"),
        "y3": ("x1", "h2"),
        "y4": ("x2", "h2"),
    }


def test_p1_le_regime_porte_la_marque_de_l_entree() -> None:
    """Une sortie oscille en regime si, et seulement si, l'entree est `x2`."""
    for entree in ("x1", "x2"):
        for reponse in ("h1", "h2"):
            oscille, _ = ap.traits(ap.sortie(entree, reponse))
            assert oscille == (entree == "x2")


def test_p1_le_transitoire_porte_la_marque_du_systeme() -> None:
    """Le transitoire dure si, et seulement si, la reponse est `h2`."""
    for entree in ("x1", "x2"):
        for reponse in ("h1", "h2"):
            _, dure = ap.traits(ap.sortie(entree, reponse))
            assert dure == (reponse == "h2")


def test_p1_la_valeur_finale_de_la_reponse_a_l_echelon() -> None:
    """Second chemin : `x1*h1` tend vers l'aire de `h1`, qui vaut un."""
    assert ap.sortie("x1", "h1")[-1] == pytest.approx(1.0, abs=1e-2)


# --- Planche 2 : le spectre d'une salve de cosinus --------------------------


def test_p2_deux_lobes_centres_sur_la_porteuse() -> None:
    """Planche 2 : deux lobes, en `+f0` et `-f0`, de hauteur `tau/2`."""
    assert sp.simplify(sx.hauteur_du_lobe() - sx.DUREE / 2) == 0
    assert sx.spectre_de_la_salve(-sx.F0) == sx.spectre_de_la_salve(sx.F0)


def test_p2_la_largeur_des_lobes() -> None:
    """Le spectre s'annule en `f0 +- 1/tau` : un lobe large de `2/tau`."""
    assert sx.spectre_de_la_salve(sx.F0 + 1 / sx.DUREE) == 0
    assert sx.spectre_de_la_salve(sx.F0 - 1 / sx.DUREE) == 0
    assert sx.largeur_du_lobe() == 2 / sx.DUREE


def test_p2_le_spectre_normalise() -> None:
    """Un au sommet du lobe, zero a la frequence nulle et au premier zero."""
    sommet, nul, zero = sx.spectre_normalise([1.0, 0.0, 1.125])
    assert sommet == pytest.approx(1)
    assert nul == pytest.approx(0, abs=1e-12)
    assert zero == pytest.approx(0, abs=1e-12)


def test_p2_presque_rien_a_la_frequence_nulle() -> None:
    """Un nombre entier de periodes : la valeur moyenne de la salve est nulle."""
    assert sx.spectre_de_la_salve(0) == 0


# --- Planche 3 : d'une onde isolee a l'onde periodique ----------------------


def test_p3_les_six_decibels_de_la_figure() -> None:
    """Planche 3 : le module vaut -6 dB a frequence nulle, l'aire etant 1/2."""
    assert sx.image_de_la_rampe(0) == sp.Rational(1, 2)
    assert sx.niveau_en_decibels(0) == pytest.approx(-6.02, abs=5e-3)


def test_p3_les_niveaux_lus_sur_la_figure() -> None:
    """Planche 3 : environ -36 dB a 10 Hz et -42 dB a 20 Hz."""
    assert sx.niveau_en_decibels(10) == pytest.approx(-36, abs=1)
    assert sx.niveau_en_decibels(20) == pytest.approx(-42, abs=1)


def test_p3_les_modules_en_serie() -> None:
    """Le meme module, calcule pour plusieurs frequences d'un coup."""
    nul, dix = sx.modules_de_la_rampe([0.0, 10.0])
    assert nul == pytest.approx(0.5)
    assert 20 * math.log10(dix) == pytest.approx(sx.niveau_en_decibels(10))


def test_p3_la_phase_plate_de_la_figure() -> None:
    """Planche 3 : +90 degres aux frequences entieres positives, -90 aux negatives."""
    assert sx.phase_en_degres(5) == pytest.approx(90)
    assert sx.phase_en_degres(-5) == pytest.approx(-90)
    assert sp.im(sx.raie_de_l_onde_periodique(10)) > 0


def test_p3_des_raies_tous_les_dixiemes_de_hertz() -> None:
    """`T0 = 10` s : une raie tous les 0,1 Hz, d'amplitude `|X(k/T0)|/T0`."""
    assert sx.ecart_des_raies() == sp.Rational(1, 10)
    assert sx.raie_de_l_onde_periodique(0) == sp.Rational(1, 20)
    assert sx.niveau_de_la_raie(0) == pytest.approx(-26.02, abs=5e-3)


def test_p3_l_enveloppe_descend_de_vingt_decibels() -> None:
    """Diviser par `T0 = 10` abaisse chaque niveau de 20 dB."""
    for rang in (0, 3, 50):
        ecart = sx.niveau_en_decibels(rang / 10) - sx.niveau_de_la_raie(rang)
        assert ecart == pytest.approx(20)


# --- Planches 13 et 18 : deux creneaux convolues ----------------------------


def test_p13_le_trapeze_du_manuscrit() -> None:
    """Planche 13 : de 0 a 6, un palier de hauteur 2 entre 2 et 4."""
    assert cv.sommets(cv.PLANCHE_13) == ((0, 0), (2, 2), (4, 2), (6, 0))


def test_p18_le_trapeze_du_manuscrit() -> None:
    """Planche 18 : de 4 a 10, un palier de hauteur 4 entre 6 et 8."""
    assert cv.sommets(cv.PLANCHE_18) == ((4, 0), (6, 4), (8, 4), (10, 0))


def test_les_debuts_s_ajoutent_et_les_hauteurs_se_multiplient() -> None:
    """Hors des planches : deux creneaux decales, de hauteurs 3 et 2."""
    autres = (cv.Creneau(1, 2, 3), cv.Creneau(3, 1, 2))
    assert cv.sommets(autres) == ((4, 0), (5, 6), (6, 6), (7, 0))
    assert cv.par_integration(autres, 5.5) == pytest.approx(6, abs=2e-2)


@pytest.mark.parametrize("donnees", [cv.PLANCHE_13, cv.PLANCHE_18])
def test_les_trapezes_par_l_integrale(donnees: cv.Creneaux) -> None:
    """Second chemin : l'integrale de convolution, echantillonnee."""
    for instant, hauteur in cv.sommets(donnees):
        assert cv.par_integration(donnees, instant) == pytest.approx(hauteur, abs=2e-2)
    milieu = (cv.sommets(donnees)[0][0] + cv.sommets(donnees)[1][0]) / 2
    attendue = cv.sommets(donnees)[1][1] / 2
    assert cv.par_integration(donnees, milieu) == pytest.approx(attendue, abs=2e-2)


@pytest.mark.parametrize("donnees", [cv.PLANCHE_13, cv.PLANCHE_18])
def test_l_aire_de_la_sortie_est_le_produit_des_aires(donnees: cv.Creneaux) -> None:
    """L'aire d'un produit de convolution est le produit des deux aires."""
    assert cv.aire_de_la_sortie(donnees) == cv.produit_des_aires(donnees)


# --- Planche 17 : la puissance d'une salve repetee --------------------------


def test_p17_la_puissance_du_manuscrit() -> None:
    """Planche 17 : `P = (4/T)(0,01/2)`, soit 2/3 pour le `T = 0,03` s du dessin."""
    assert sx.puissance(sp.Rational(3, 100)) == sp.Rational(2, 3)


def test_p17_la_periode_pour_une_puissance_unite() -> None:
    """Planche 17 : `T = 2 x 0,01 = 0,02` s ; la valeur efficace y vaut un."""
    assert sx.periode_pour_une_puissance_unite() == sp.Rational(1, 50)
    assert math.isclose(float(sx.valeur_efficace(sp.Rational(1, 50))), 1.0)


def test_p17_le_facteur_sur_la_periode() -> None:
    """Planche 17 : diviser la valeur efficace par 10 multiplie `T` par 100."""
    assert sx.facteur_pour_diviser_la_valeur_efficace(10) == 100
