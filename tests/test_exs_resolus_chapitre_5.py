"""Signaux et systemes : les exercices resolus, chapitre 5 (0.29 a 0.35).

Source : `SignauxSystemes/EXS/Exs_01_02_04_05_Resolus.pdf`, pages 21 a 26.
Les solutions sont recopiees comme oracles et recoupees. Elles sont
justes, mais leur numerotation glisse a la fin du chapitre : celle qui
est titree 0.34 complete 0.29, celle titree 0.35 repond a 0.34, et celle
titree 0.36 a 0.35.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.signaux.application import resolus_regimes as rg
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

# --- 0.29 : l'allure sans inverser Laplace ----------------------------------


def test_0_29_la_solution() -> None:
    """Page 23 : « periode 2 pi s, amorti apres 5 s : meme pas une periode »."""
    duree, periode = rg.duree_et_periode_de_la_tension()
    assert (duree, periode) == (5, 2 * sp.pi)
    assert duree < periode


def test_0_29_l_expression_titree_0_34() -> None:
    """Page 24 : `racine(2) e^(-t) cos(t - theta)`, `theta = arctan(-1) = -pi/4`."""
    imprimee = sp.sqrt(2) * sp.exp(-T) * sp.cos(T + sp.pi / 4)
    assert sp.simplify(rg.tension_de_la_capacite() - imprimee) == 0
    assert sp.atan(-1) == -sp.pi / 4


# --- 0.30 : l'enclenchement d'un circuit R-L sur une tension sinusoidale ----


def test_0_30_le_courant_de_la_solution_verifie_le_circuit() -> None:
    """Page 23 : regime moins transitoire ; il verifie la maille et part de zero."""
    assert rg.residu_de_la_maille() == 0
    assert sp.simplify(rg.courant().subs(T, 0)) == 0


def test_0_30_l_admittance_de_la_solution() -> None:
    """Page 23 : `Y e^(j theta) = 1/(j w L + R)`."""
    module, argument = rg.admittance()
    attendue = 1 / (sp.I * rg.W * rg.L + rg.R)
    assert sp.simplify(module - sp.Abs(attendue)) == 0
    assert sp.simplify(sp.tan(argument) + rg.W * rg.L / rg.R) == 0


def test_0_30_la_crete_peut_atteindre_deux_fois_le_regime() -> None:
    """Page 23 : « deux fois la valeur de regime pour phi + theta = k pi »."""
    assert rg.crete_relative(constante_en_periodes=1000) == pytest.approx(2, abs=1e-3)
    assert rg.crete_relative(constante_en_periodes=0.5) == pytest.approx(
        1 + math.exp(-1)
    )
    assert rg.phase_du_pire_cas() == -rg.admittance()[1]


def test_0_30_sans_transitoire() -> None:
    """Second chemin : pour `phi + theta = pi/2` le transitoire disparait."""
    assert rg.amplitude_du_transitoire(sp.pi / 2 - rg.admittance()[1]) == 0


# --- 0.31 a 0.34 : les exercices de la seance « Chap. 4-5 » -----------------


def test_0_31_la_condition_de_la_solution() -> None:
    """Page 23 : `R > (1/2) racine(L/C)`."""
    assert sp.simplify(rg.resistance_limite() ** 2 - rg.L / (4 * rg.C)) == 0


def test_0_32_la_transmittance_et_l_impulsionnelle() -> None:
    """Page 24 : `H = (R/L)/(p + R/L) = 10^4/(p + 10^4)`, `tau = 10^-4` s."""
    attendue = (rg.R / rg.L) / (P + rg.R / rg.L)
    assert sp.simplify(rg.transmittance_sans_le_condensateur() - attendue) == 0
    depart, constante = rg.reperes_de_l_impulsionnelle()
    assert depart == pytest.approx(1e4)
    assert constante == pytest.approx(1e-4)


def test_0_32_les_reponses_ne_se_chevauchent_pas() -> None:
    """Page 24 : « en pratique 10^-2 >> 10^-4 » : cent constantes de temps."""
    assert rg.periode_du_train_en_constantes() == pytest.approx(100)


def test_0_33_la_condition_de_la_solution() -> None:
    """Page 24 : `sigma >= rho`, soit `alpha >= 20 racine(2)`."""
    assert rg.alpha_limite() == 20 * sp.sqrt(2)


def test_0_34_les_limites_titrees_0_35() -> None:
    """Page 25 : `a(0) = 0` et `a(infini) = 1/2` ; un sommet a 0,52."""
    assert rg.limites_de_l_indicielle() == (0, sp.Rational(1, 2))
    assert rg.sommet_de_l_indicielle() == pytest.approx(0.52, abs=5e-3)


# --- 0.35 : une entree periodique dans un passe-bas -------------------------


def test_0_35_la_reponse_en_frequence() -> None:
    """Page 26 : `H(w) = 1/(1 + j w)`, la transformee de `e^(-t)` en `p = j w`."""
    assert sp.simplify(rg.reponse_en_frequence(rg.W) - 1 / (1 + sp.I * rg.W)) == 0


def test_0_35_les_coefficients_titres_0_36() -> None:
    """Page 26 : `Y_k = 1/(1 + jk) x 1/(1 + j k w0)`, avec `w0 = 20 pi`."""
    k = sp.Symbol("k", integer=True)
    attendu = 1 / ((1 + sp.I * k) * (1 + sp.I * k * 20 * sp.pi))
    assert sp.simplify(rg.coefficient_de_sortie(k) - attendu) == 0
    assert sp.simplify(rg.PULSATION_FONDAMENTALE - 20 * sp.pi) == 0


def test_0_35_la_composante_continue_passe_et_le_reste_s_ecrase() -> None:
    """`Y_0 = 1` ; le fondamental est divise par 63 environ."""
    assert rg.coefficient_de_sortie(0) == 1
    assert rg.attenuation_du_fondamental() == pytest.approx(62.8, abs=5e-2)
