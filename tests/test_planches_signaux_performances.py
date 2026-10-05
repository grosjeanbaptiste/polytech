"""Planches de signaux et systemes : les performances d'une boucle fermee.

Source : `SignauxSystemes/SLIDES/Slides_08_Performances.pdf` (45 planches),
lues sur les pages rendues.

Les oracles : les tableaux des erreurs de regime (planches 181 et 182),
les trois exemples des planches 183 a 185, et les illustrations des
planches 191 a 212 -- `|M|max = 2,3 dB` pour `racine(2) / (p (1 + p))`,
puis `-0,83 dB`, `4 dB`, `5 rad/s`, 0,91, 38 %, 1,246 et 0,4 s pour
`10 / (1 + p)^2`, et `0,79 rad/s` pour l'exemple a delai.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.signaux.domain.bode import classe_du_systeme
from polytech.signaux.domain.boucle_fermee import Boucle
from polytech.signaux.domain.erreurs_de_regime import (
    erreur_de_consigne,
    erreur_de_perturbation,
    sortie_ideale,
)
from polytech.signaux.domain.fonction_m import (
    CaracteristiquesDeM,
    caracteristiques_de_m,
    depassement_estime,
    fonction_m,
    temps_de_montee_estime,
    valeur_finale_estimee,
)
from polytech.signaux.domain.performances import (
    erreur_statique_de_consigne,
    module_en_continu,
)
from polytech.signaux.domain.poles_numeriques import reponse_indicielle
from polytech.signaux.domain.reponse_indicielle import (
    depassement_du_second_ordre,
    temps_de_montee_depuis_la_bande,
)
from polytech.signaux.domain.stabilite import Stabilite
from polytech.signaux.domain.symboles import GAIN, IMAGE
from polytech.signaux.domain.temps_caracteristiques import temps_de_montee
from polytech.signaux.domain.transmittance import Transmittance

P, K = IMAGE, GAIN


def test_l_erreur_de_consigne_est_l_ecart_a_la_sortie_ideale() -> None:
    """Planches 175 et 176 : `E_C = (T_R - 1 / G_s) R = -R / (G_s (1 + L))`."""
    c, g1, g2, gs, consigne = sp.symbols("C G_1 G_2 G_s R")
    boucle = Boucle(correcteur=c, actionneur=g1, procede=g2, capteur=gs)
    assert sortie_ideale(boucle, consigne) == consigne / gs
    reelle = boucle.vis_a_vis_de_la_consigne() * consigne
    ecart = reelle - sortie_ideale(boucle, consigne)
    assert sp.simplify(ecart + consigne / (gs * (1 + gs * c * g1 * g2))) == 0


def test_l_exemple_1_sans_integrateur_les_deux_erreurs_sont_finies() -> None:
    """Planche 183 : `C = 2`, `R = 5 / p`, `P = 10 / p` ; la sortie finit a 4,55."""
    boucle = Boucle(2, 5 / (1 + P), 1 / (1 + 5 * P), 1)
    assert classe_du_systeme(boucle.ouverte()) == 0
    consigne = erreur_de_consigne(boucle, 5)
    assert consigne == -sp.Rational(5, 11)
    assert float(5 + consigne) == pytest.approx(4.55, abs=0.005)
    assert erreur_de_perturbation(boucle, 10) == -sp.Rational(10, 11)
    assert float(consigne) == pytest.approx(-5 * erreur_statique_de_consigne(10.0))


def test_l_exemple_2_un_integrateur_avant_la_perturbation_annule_tout() -> None:
    """Planche 184 : `G_1 = 1 / p`, `G_s = 4` ; sortie ideale 1,25."""
    boucle = Boucle(sp.Rational(1, 10), 1 / P, 1 / (1 + 5 * P), 4)
    assert classe_du_systeme(boucle.ouverte()) == 1
    assert sortie_ideale(boucle, 5) == sp.Rational(5, 4)
    assert erreur_de_consigne(boucle, 5) == 0
    assert erreur_de_perturbation(boucle, 10) == 0


def test_un_integrateur_dans_le_procede_laisse_une_erreur_de_perturbation() -> None:
    """Planche 180 : `e_P = -P^0 / (G_s C(0) G_1(0))`, mais `e_C = 0`."""
    vingtieme = sp.Rational(1, 20)
    boucle = Boucle(vingtieme, 5 / (1 + P), 1 / (P * (1 + P)), 4)
    assert boucle.est_stable()
    assert erreur_de_consigne(boucle, 5) == 0
    assert erreur_de_perturbation(boucle, 10) == -10 / (4 * vingtieme * 5)


def test_une_boucle_instable_n_a_pas_d_erreur_de_regime() -> None:
    """Planche 176 : « en regime etabli, s'il existe... : stabilite ! »."""
    boucle = Boucle(10, 5 / (1 + P), 1 / (P * (1 + P)), 4)
    with pytest.raises(ValueError, match="boucle instable"):
        erreur_de_consigne(boucle, 5)
    with pytest.raises(ValueError, match="boucle instable"):
        erreur_de_perturbation(boucle, 10)


SORTIES = [(0.1, 0.0714), (0.5, 0.1667), (1, 0.2), (2, 0.2222)]


@pytest.mark.parametrize(("gain", "finale"), SORTIES)
def test_l_exemple_3_l_erreur_diminue_quand_le_gain_augmente(
    gain: float, finale: float
) -> None:
    """Planche 185 : `G_s = 4`, sortie ideale 0,25, sortie `K / (1 + 4 K)`."""
    boucle = Boucle(sp.nsimplify(gain), 1 / (1 + P) ** 2, 1 / (1 + 5 * P), 4)
    sortie = sortie_ideale(boucle, 1) + erreur_de_consigne(boucle, 1)
    assert float(sortie) == pytest.approx(finale, abs=5e-5)
    assert float(sortie) < 0.25


def test_l_exemple_3_a_un_gain_maximal() -> None:
    """« Il existe un K_max » : `5 p^3 + 11 p^2 + 7 p + 1 + 4 K`, soit `K < 3,6`."""
    ouverte = 4 * K / ((1 + P) ** 2 * (1 + 5 * P))
    assert Stabilite(Transmittance(ouverte)).gain_critique(K) == sp.Rational(18, 5)


def test_la_fonction_m_est_l_image_de_la_boucle_fermee() -> None:
    """Planche 187 : `M = G_s C G_1 G_2 / (1 + G_s C G_1 G_2)`, `T_R = M / G_s`."""
    c, g1, g2, gs = sp.symbols("C G_1 G_2 G_s")
    boucle = Boucle(correcteur=c, actionneur=g1, procede=g2, capteur=gs)
    assert (
        sp.simplify(
            fonction_m(boucle.ouverte()) - gs * boucle.vis_a_vis_de_la_consigne()
        )
        == 0
    )


def test_le_module_de_m_en_continu_dit_la_classe() -> None:
    """Planche 196 : inferieur a un en classe 0, egal a un des la classe 1."""
    classe_zero = caracteristiques_de_m(10 / (1 + P) ** 2)
    assert classe_zero.en_continu == pytest.approx(module_en_continu(10.0))
    assert classe_zero.en_continu_en_decibels() == pytest.approx(-0.83, abs=0.005)
    classe_un = caracteristiques_de_m(sp.sqrt(2) / (P * (1 + P)))
    assert classe_un.en_continu == pytest.approx(1.0, abs=1e-6)


def test_la_valeur_finale_se_lit_sur_m_au_gain_du_capteur_pres() -> None:
    """Planche 201 : `y(infini) = |M(0)| R^0 / G_s`, sur l'exemple 3 a `K = 1`."""
    boucle = Boucle(1, 1 / (1 + P) ** 2, 1 / (1 + 5 * P), 4)
    m = caracteristiques_de_m(boucle.ouverte())
    estimee = valeur_finale_estimee(m, consigne=1.0, capteur=4.0)
    assert estimee == pytest.approx(0.2, abs=1e-6)
    exacte = sortie_ideale(boucle, 1) + erreur_de_consigne(boucle, 1)
    assert estimee == pytest.approx(float(exacte), abs=1e-6)


def test_l_exemple_1_des_abaques_touche_le_lieu_iso_m_de_2_3_db() -> None:
    """Planche 191 : `racine(2) / (p (1 + p))`."""
    m = caracteristiques_de_m(sp.sqrt(2) / (P * (1 + P)))
    assert m.maximum_en_decibels() == pytest.approx(2.35, abs=0.01)
    assert m.minimum is None
    assert isinstance(m, CaracteristiquesDeM)


def test_l_illustration_1_le_systeme_faiblement_amorti() -> None:
    """Planche 205 : `10 / (1 + p)^2`, `G_s = 1`, echelon unite.

    Estime : 0,91, 38 %, 1,256, entre 0,37 et 0,56 s. Exact : 0,91, 1,246
    et 0,4 s.
    """
    m = caracteristiques_de_m(10 / (1 + P) ** 2)
    assert m.maximum_en_decibels() == pytest.approx(4.0, abs=0.03)
    assert m.bande_passante == pytest.approx(5.0, abs=0.2)
    finale = valeur_finale_estimee(m, consigne=1.0, capteur=1.0)
    assert finale == pytest.approx(0.91, abs=0.005)
    ecart = m.maximum_en_decibels() - m.en_continu_en_decibels()
    assert ecart == pytest.approx(4.83, abs=0.03)
    assert depassement_estime(4.83) == pytest.approx(38.0, abs=0.05)
    assert finale * (1 + depassement_estime(4.83) / 100) == pytest.approx(
        1.256, abs=0.002
    )
    bas, haut = temps_de_montee_estime(0.8)
    assert (bas, haut) == pytest.approx((0.375, 0.5625))
    instants = np.linspace(0, 8, 80_001)
    reponse = reponse_indicielle([10.0], [1.0, 2.0, 11.0], instants)
    assert reponse.max() == pytest.approx(1.246, abs=5e-4)
    assert reponse[-1] == pytest.approx(0.909, abs=1e-3)
    exact = temps_de_montee(instants, reponse, regime=10 / 11)
    assert exact == pytest.approx(0.4, abs=0.02)
    assert bas < exact < haut


def test_la_regle_du_depassement_approche_le_second_ordre() -> None:
    """Planche 202 : `D(%) = 9 + 6 (|M|max - |M(0)|)` en dB, contre l'exact 37 %."""
    amortissement = 1 / sp.sqrt(11)
    exact = 100 * float(depassement_du_second_ordre(amortissement))
    assert exact == pytest.approx(37.0, abs=0.05)
    assert depassement_estime(4.83) == pytest.approx(exact, abs=1.5)
    with pytest.raises(ValueError, match="ecart negatif"):
        depassement_estime(-1.0)


def test_le_temps_de_montee_s_encadre_par_la_bande_passante() -> None:
    """Planche 201 : `0,3 < t_m BP(Hz) < 0,45`. Planches 206, 207 et 210."""
    for pulsation, bornes in (
        (1.6, (1.18, 1.77)),
        (0.77, (2.45, 3.67)),
        (1.15, (1.64, 2.46)),
    ):
        hertz = pulsation / (2 * math.pi)
        assert temps_de_montee_estime(hertz) == pytest.approx(bornes, abs=0.006)
        en_pulsation = temps_de_montee_depuis_la_bande(pulsation)
        assert en_pulsation == pytest.approx(temps_de_montee_estime(hertz), rel=0.01)
    with pytest.raises(ValueError, match="bande passante nulle ou negative"):
        temps_de_montee_estime(0.0)


def test_l_illustration_4_le_systeme_a_delai() -> None:
    """Planche 210 : `|M|max = 3,8 dB` donne 32 %, puis `y_max = 1,32`."""
    assert depassement_estime(3.8) == pytest.approx(31.8)
    assert 1 + depassement_estime(3.8) / 100 == pytest.approx(1.32, abs=0.003)


EXEMPLE_A_DELAI = 0.7 * (1 + P) * sp.exp(-P / 2) / (P * (1 + P / 10) ** 2)
"""Planche 193 : `0,7 (1 + p) exp(-0,5 p) / (p (1 + 0,1 p)^2)`."""


def test_l_exemple_3_des_abaques_a_un_creux_puis_un_pic() -> None:
    """Planches 193 et 212 : `|M|min`, puis un pic a 4 dB ; `BP = 0,79 rad/s`."""
    m = caracteristiques_de_m(EXEMPLE_A_DELAI)
    assert m.en_continu == pytest.approx(1.0, abs=1e-4)
    assert m.minimum is not None
    assert 20 * math.log10(m.minimum) == pytest.approx(-3.7, abs=0.05)
    assert m.maximum_en_decibels() == pytest.approx(4.0, abs=0.05)
    assert m.bande_passante == pytest.approx(0.79, abs=0.005)
    bas, haut = temps_de_montee_estime(m.bande_passante / (2 * math.pi))
    assert (bas, haut) == pytest.approx((2.4, 3.6), abs=0.01)


def test_la_legende_de_la_planche_211_annonce_un_delai_d_une_seconde() -> None:
    """« avec tau = 1 s » : les courbes sont celles de `tau = 0,5 s`.

    Avec une seconde, la bande passante vaudrait 3,3 rad/s et le pic
    8,6 dB, et non 0,79 rad/s et 4 dB.
    """
    une_seconde = 0.7 * (1 + P) * sp.exp(-P) / (P * (1 + P / 10) ** 2)
    m = caracteristiques_de_m(une_seconde)
    assert m.bande_passante == pytest.approx(3.33, abs=0.01)
    assert m.maximum_en_decibels() == pytest.approx(8.6, abs=0.05)
