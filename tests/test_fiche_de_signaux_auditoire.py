"""Fiche de signaux : les 128 etudiants bruyants (juin 2018, question 3).

Source : `Exam_2018_Juin_Enonce.pdf`, question 3 A, B et C.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.maths.domain.comparaison import egales, proches
from polytech.signaux.domain.auditoire import (
    etudiants_a_garder,
    facteur_d_amplitude,
    facteur_pour_atteindre,
    rsb_avec,
)
from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.coup_de_belier import (
    graphe_du_coup_de_belier,
    reglage_pour_un_etablissement,
)
from polytech.signaux.domain.erreurs_de_regime import (
    erreur_de_consigne,
    erreur_de_consigne_selon_la_fiche,
)
from polytech.signaux.domain.symboles import IMAGE as P
from tests.fiches_toml import articles


def test_la_question_a_demande_de_faire_sortir_soixante_quatre_etudiants() -> None:
    """A : de 30 a 33 dB, le bruit est divise par 2 : 64 etudiants restent."""
    assert etudiants_a_garder(128, 30.0, 33.0) == 64


def test_la_question_b_un_seul_etudiant_donne_cinquante_et_un_decibels() -> None:
    """B : `30 + 10 log 128`."""
    assert proches(rsb_avec(1, de_reference=(128, 30.0)), 51.07, 0.01)


def test_la_question_c_le_facteur_ne_depend_pas_du_nombre_d_etudiants() -> None:
    """C : 3 dB d'amplitude, `sqrt 2`, pour 100 comme pour 128 etudiants."""
    facteur = facteur_pour_atteindre(30.0, 33.0)
    assert proches(facteur, 1.4125, 1e-3)
    assert proches(facteur_d_amplitude(10 * math.log10(2)), math.sqrt(2), 1e-12)
    for restants in (100, 128):
        base = rsb_avec(restants, de_reference=(restants, 30.0))
        corrige = rsb_avec(restants, de_reference=(restants, 30.0), amplitude=facteur)
        assert proches(corrige - base, 3.0, 1e-9)


def test_le_piege_cent_ou_cent_vingt_huit_change_le_facteur_demande() -> None:
    """Les deux nombres ne s'accordent pas : 100 donne deja 31,07 dB."""
    deja = rsb_avec(100, de_reference=(128, 30.0))
    assert proches(deja, 31.07, 0.01)
    assert proches(facteur_pour_atteindre(deja, 33.0), 1.2495, 1e-3)


def test_le_coup_de_belier_ne_saute_pas_et_s_etablit_en_trois_periodes() -> None:
    """Juin 2022 (1) : `h(0+) = 0` ; `5 tau = 3 T` fixe `alpha` pour `beta` donne."""
    alpha, beta = sp.Rational(1, 2), sp.Integer(2)
    h = graphe_du_coup_de_belier(alpha, beta)
    assert egales(h, beta / (P**2 + alpha * P + 2))
    assert sp.limit(P * h, P, sp.oo) == 0
    reglage = reglage_pour_un_etablissement(3)
    tau, periode = 2 / reglage, 2 * sp.pi / sp.sqrt(2 - reglage**2 / 4)
    assert proches(float(5 * tau), float(3 * periode), 1e-9)


def test_l_erreur_de_consigne_du_cours_est_l_oppose_de_celle_de_la_lib() -> None:
    """Fiche : `e_C = Y_i - y` ; la lib rend `y - Y_i`."""
    boucle = Boucle(correcteur=1, actionneur=1, procede=4 / (P + 1), capteur=1)
    assert erreur_de_consigne_selon_la_fiche(boucle, 1) == -erreur_de_consigne(
        boucle, 1
    )
    assert erreur_de_consigne_selon_la_fiche(boucle, 1) == sp.Rational(1, 5)


def test_la_fiche_ecrit_la_convention_du_cours_pour_l_erreur() -> None:
    formule = articles("SignauxSystemes")["erreurs_de_sortie_en_regime"]["formule"]
    assert r"e_C = Y_i - y(\infty)" in formule
