"""Fiche de signaux : erreurs de regime, fonction M, estimation de la reponse.

Articles « erreurs », « fonction M » et « estimer » de
`cursus/fiches/SignauxSystemes.toml`.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

from polytech.maths.domain.comparaison import egales, proches
from polytech.maths.domain.laplace.theoremes import valeur_finale
from polytech.maths.domain.symboles import IMAGE as Q
from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.erreurs_de_regime import (
    erreur_de_consigne,
    erreur_de_perturbation,
)
from polytech.signaux.domain.fonction_m import (
    caracteristiques_de_m,
    depassement_estime,
    fonction_m,
    valeur_finale_estimee,
)
from polytech.signaux.domain.poles_numeriques import reponse_indicielle
from polytech.signaux.domain.reponse_indicielle import (
    bande_passante_du_second_ordre,
    depassement_du_second_ordre,
    temps_de_montee_depuis_la_bande,
)
from polytech.signaux.domain.second_ordre import SecondOrdre
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.temps_caracteristiques import temps_de_montee

C, G1, G2, GS = sp.symbols("C G_1 G_2 G_s")
R0, P0 = sp.symbols("R_0 P_0")


def test_la_sortie_de_regime_et_l_erreur_de_consigne_du_cours() -> None:
    """Article « Les erreurs en regime » : `y(oo) = T_R(0) R_0 + T_P(0) P_0`.

    Le cours ecrit l'erreur `Y_i - y` ; la bibliotheque rend `y - Y_i`.
    """
    boucle = Boucle(correcteur=1, actionneur=1, procede=4 / (P + 1), capteur=2)
    sortie = boucle.sortie(R0, P0)
    regime = valeur_finale(sortie.subs(P, Q) / Q)
    t_r, t_p = boucle.vis_a_vis_de_la_consigne(), boucle.vis_a_vis_de_la_perturbation()
    assert egales(regime, t_r.subs(P, 0) * R0 + t_p.subs(P, 0) * P0)
    k_bo = boucle.ouverte().subs(P, 0)
    ideale = R0 / 2
    assert egales(ideale - t_r.subs(P, 0) * R0, ideale / (1 + k_bo))
    assert egales(erreur_de_consigne(boucle, R0), -(ideale / (1 + k_bo)))


def test_un_integrateur_dans_la_boucle_annule_l_erreur_de_consigne() -> None:
    """Article « Les erreurs en regime » : classe 1, erreur nulle."""
    en_amont = Boucle(correcteur=1 / P, actionneur=1, procede=4 / (P + 1), capteur=1)
    en_aval = Boucle(correcteur=1, actionneur=1, procede=4 / (P * (P + 1)), capteur=1)
    assert erreur_de_consigne(en_aval, R0) == 0
    assert erreur_de_perturbation(en_amont, P0) == 0
    assert erreur_de_perturbation(en_aval, P0) == -P0


def test_la_fonction_m_donne_le_regime_et_la_bande() -> None:
    """Article « La fonction M » : `|M(0)| = K/(1+K)` en classe 0, `1` en classe 1."""
    classe0, classe1 = 4 / (P + 1), 4 / (P * (P + 1))
    assert egales(fonction_m(classe0).subs(P, 0), sp.Rational(4, 5))
    assert proches(caracteristiques_de_m(classe0).en_continu, 0.8, 1e-6)
    assert proches(caracteristiques_de_m(classe1).en_continu, 1.0, 1e-6)
    assert egales(fonction_m(classe1).subs(P, 0), 1)


def test_le_module_de_m_de_la_figure_est_celui_du_second_ordre() -> None:
    """Figure : `1 / sqrt((1 - x^2)^2 + (2 zeta x)^2)`."""
    w = sp.Symbol("w", positive=True)
    for zeta in (sp.Rational(3, 10), sp.Rational(7, 10)):
        h = SecondOrdre(gain=1, constante=1, amortissement=zeta).de_bode()
        module = sp.Abs(h.expression.subs(P, sp.I * w))
        attendu = 1 / sp.sqrt((1 - w**2) ** 2 + (2 * zeta * w) ** 2)
        assert egales(module, attendu)


def test_la_valeur_finale_estimee_divise_par_le_gain_du_capteur() -> None:
    """Juin 2022 (4) : un capteur de gain 5 divise la sortie de regime par 5."""
    ouverte = 5 * 4 / (P + 1)
    m = caracteristiques_de_m(ouverte)
    attendu = float(sp.Rational(20, 21)) * 3 / 5
    assert proches(valeur_finale_estimee(m, consigne=3, capteur=5), attendu, 1e-9)


def test_le_depassement_estime_approche_l_exact_du_second_ordre() -> None:
    """Article « Estimer la reponse indicielle » : `D(%) ~ 9 + 6 (Mmax - M0)`."""
    zeta = 0.3
    ouverte = 1 / (P * (P + 2 * sp.Float(zeta)))
    m = caracteristiques_de_m(ouverte)
    exact = 100 * float(depassement_du_second_ordre(sp.Float(zeta)))
    estime = depassement_estime(m.maximum_en_decibels() - m.en_continu_en_decibels())
    assert proches(estime, exact, 3.0)


def test_le_temps_de_montee_est_encadre_par_la_bande_passante_en_hertz() -> None:
    """Article « Estimer la reponse indicielle » : `0,3 < t_m BP < 0,45`."""
    for zeta in (0.3, 0.5, 0.7, 1.0):
        instants = np.linspace(0, 30, 30001)
        reponse = reponse_indicielle([1], [1, 2 * zeta, 1], instants)
        montee = temps_de_montee(instants, reponse, regime=1.0)
        bande = bande_passante_du_second_ordre(1.0, zeta) / (2 * math.pi)
        bas, haut = temps_de_montee_depuis_la_bande(2 * math.pi * bande)
        assert bas < montee < haut
