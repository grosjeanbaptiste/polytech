"""Le TP 1 de signaux : chaque reponse recoupee par un chemin independant.

Le TP n'a pas de corrige officiel : c'est le devoir « TP1 Devoir » du
cours de Charleroi, et son enonce dit qu'il n'est pas cote. Les oracles
sont donc ceux du cours, et non ceux d'un manuscrit :

- Q1 et Q2 : la deconvolution lit `b` sans le deviner ;
- Q3 : `T sinc(T f)` par le changement d'echelle ;
- Q4 : `p = -R/(2L) +- j racine(1/(LC) - (R/(2L))^2)`, a la main ;
- Q5 : le critere de Routh, contre les racines numeriques ;
- Q6 : un pole dominant annule le denominateur, et c'est le plus lent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polytech.signaux.application import enonce_lab_01 as e
from polytech.signaux.domain.echantillons import convoluer, deconvoluer, peigne, porte
from polytech.signaux.domain.modeles import (
    coefficients,
    resistance_critique,
    transmittance_d_une_equation,
    transmittance_lc_serie_sur_r,
)
from polytech.signaux.domain.poles_numeriques import (
    constante_de_temps,
    est_stable,
    frequence_d_oscillation,
    poles_dominants,
    poles_et_zeros,
    routh_du_troisieme_degre,
)


def test_q1_le_triangle_se_deconvolue_en_la_porte_elle_meme() -> None:
    a = porte(e.UN_LARGEUR, e.UN_SUPPORT)
    c = convoluer(a, porte(e.UN_LARGEUR, e.UN_LARGEUR))
    assert np.allclose(deconvoluer(c, a), np.ones(e.UN_LARGEUR))


def test_q1_le_triangle_culmine_a_dix_au_dixieme_echantillon() -> None:
    c = convoluer(porte(e.UN_LARGEUR, e.UN_SUPPORT), porte(e.UN_LARGEUR, e.UN_LARGEUR))
    assert (c.max(), int(c.argmax())) == (10.0, 9)


def test_q2_le_peigne_periodise_la_porte() -> None:
    a = porte(e.DEUX_LARGEUR, e.DEUX_SUPPORT)
    c = convoluer(a, peigne(e.DEUX_PERIODE, e.DEUX_IMPULSIONS))
    attendu = np.tile(a, e.DEUX_IMPULSIONS)
    assert np.allclose(c[: len(attendu)], attendu)


def test_q4_les_poles_sont_ceux_de_la_formule_du_second_ordre() -> None:
    num, den = coefficients(
        transmittance_lc_serie_sur_r(e.QUATRE_R, e.QUATRE_L, e.QUATRE_C)
    )
    poles, zeros, _ = poles_et_zeros(num, den)
    amortissement = e.QUATRE_R / (2 * e.QUATRE_L)
    pseudo = math.sqrt(1 / (e.QUATRE_L * e.QUATRE_C) - amortissement**2)
    assert np.allclose(sorted(poles.imag), [-pseudo, pseudo])
    assert np.allclose(poles.real, -amortissement)
    assert np.allclose(zeros, 0.0)


def test_q4_constante_de_temps_et_frequence() -> None:
    pole = complex(-e.QUATRE_R / (2 * e.QUATRE_L), 3122.4989992)
    assert constante_de_temps(pole) == pytest.approx(2e-3)
    assert frequence_d_oscillation(pole) == pytest.approx(496.96, abs=0.01)


def test_q4f_la_resistance_choisie_depasse_la_critique() -> None:
    critique = float(resistance_critique(e.QUATRE_L, e.QUATRE_C))
    assert critique == pytest.approx(6324.555, abs=1e-3)
    assert critique < e.QUATRE_R_CHOISIE


def test_q4f_la_resistance_choisie_donne_deux_poles_reels_distincts() -> None:
    num, den = coefficients(
        transmittance_lc_serie_sur_r(e.QUATRE_R_CHOISIE, e.QUATRE_L, e.QUATRE_C)
    )
    poles, _, _ = poles_et_zeros(num, den)
    assert np.allclose(poles.imag, 0.0)
    assert abs(poles[0] - poles[1]) > 1.0


def test_q5_la_capsule_est_instable_par_routh() -> None:
    assert routh_du_troisieme_degre([float(c) for c in e.CINQ_SORTIE]) < 0


def test_q5_la_capsule_est_instable_par_ses_racines() -> None:
    num, den = coefficients(transmittance_d_une_equation(e.CINQ_SORTIE, e.CINQ_ENTREE))
    poles, _, _ = poles_et_zeros(num, den)
    assert not est_stable(poles)
    assert int(np.sum(poles.real > 0)) == 2


def test_q6_les_poles_dominants_annulent_le_denominateur() -> None:
    poles, _, _ = poles_et_zeros(e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR)
    for pole in poles_dominants(poles):
        assert abs(np.polyval(e.SIX_DENOMINATEUR, pole)) < 1e-9


def test_q6_les_poles_dominants_oscillent_a_un_hertz_environ() -> None:
    poles, _, _ = poles_et_zeros(e.SIX_NUMERATEUR, e.SIX_DENOMINATEUR)
    dominant = poles_dominants(poles)[0]
    assert frequence_d_oscillation(dominant) == pytest.approx(1.0794, abs=1e-4)
    assert constante_de_temps(dominant) == pytest.approx(0.9818, abs=1e-4)
