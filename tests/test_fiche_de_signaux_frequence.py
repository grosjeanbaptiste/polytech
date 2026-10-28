"""Fiche de signaux : regime harmonique, Bode, retard, marges.

Articles du bloc frequentiel de `cursus/fiches/SignauxSystemes.toml`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.maths.domain.comparaison import egales, proches
from polytech.signaux.domain.bode import courbe_du_delai
from polytech.signaux.domain.harmonique import (
    AnalyseHarmonique,
    en_decibels,
    influence_du_gain,
    parties_reelle_et_imaginaire,
)
from polytech.signaux.domain.marges import marges_de_stabilite
from polytech.signaux.domain.stabilite import (
    marge_de_phase_avec_retard,
    phase_ajoutee_par_un_retard,
)
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import PULSATION as W
from polytech.signaux.domain.trace_asymptotique import trace_d_une_transmittance
from polytech.signaux.domain.transmittance import Transmittance

TEMPS = sp.Symbol("t", real=True)
BODE = 20 * P * (P + sp.Rational(1, 2)) / ((10 * P + 1) * (P + 1))


def test_la_sortie_de_regime_est_a_fois_le_module_dephasee_de_l_argument() -> None:
    """Article « Le regime harmonique » : `A |H| cos(w t + arg H)` resout l'equation."""
    omega, amplitude = sp.Rational(1, 10), 5
    analyse = AnalyseHarmonique(Transmittance(expression=1 / (1 + P)))
    sortie = (
        amplitude
        * analyse.module(omega)
        * sp.cos(omega * TEMPS + analyse.argument(omega))
    )
    residu = sp.diff(sortie, TEMPS) + sortie - amplitude * sp.cos(omega * TEMPS)
    assert all(proches(float(residu.subs(TEMPS, x)), 0, 1e-12) for x in (0.3, 2.0))


def test_la_courbe_de_nyquist_de_la_figure_est_celle_de_deux_sur_un_plus_jw() -> None:
    """Figure : `(2 / (1 + w^2), -2 w / (1 + w^2))`."""
    reel, imag = parties_reelle_et_imaginaire(2 / (1 + sp.I * W))
    assert egales(reel, 2 / (1 + W**2))
    assert egales(imag, -2 * W / (1 + W**2))


def test_les_asymptotes_de_bode_de_la_figure_et_les_pentes() -> None:
    """Article « Bode d'ordre quelconque » : cassures `0,1`, `0,5`, `1`."""
    trace = trace_d_une_transmittance(BODE)
    assert [c for c, _ in trace.cassures] == [0.1, 0.5, 1.0]
    assert trace.pentes()[0] == 20
    for w in (0.01, 0.3, 0.7, 5.0):
        figure = 20 * w * max(w, 0.5) / (max(10 * w, 1) * max(w, 1))
        assert proches(trace.gain_en_decibels(w), 20 * math.log10(figure), 1e-9)
        exact = AnalyseHarmonique(Transmittance(expression=BODE)).module(sp.Float(w))
        assert proches(float(en_decibels(exact)), float(en_decibels(figure)), 20)


def test_chaque_integrateur_ajoute_moins_vingt_decibels_par_decade() -> None:
    """Article « Bode d'ordre quelconque » : `K / p^alpha`, pente `-20 alpha`."""
    for alpha in (1, 2, 3):
        pentes = trace_d_une_transmittance(4 / (P**alpha * (P + 1))).pentes()
        assert pentes[0] == -20 * alpha
        assert pentes[-1] == -20 * (alpha + 1)


def test_un_gain_deplace_le_module_et_un_delai_ne_touche_que_la_phase() -> None:
    """Article « Gain et delai » : `+20 log K`, puis `arg - w tau`."""
    base = (1 / sp.sqrt(1 + W**2), -sp.atan(W))
    module, argument = influence_du_gain(base, 10)
    assert egales(en_decibels(module) - en_decibels(base[0]), 20)
    assert egales(argument, base[1])
    gain_du_delai, phase_du_delai = courbe_du_delai(2)
    assert egales(gain_du_delai, 1)
    assert egales(phase_du_delai, -2 * W)


def test_la_courbe_de_black_de_la_figure_et_le_delai_de_deux_secondes() -> None:
    """Figure : `L = 1/(p(p+1))`, phase `-90 - atan w`, gain `1/(w sqrt(1+w^2))`."""
    ouverte = Transmittance(expression=1 / (P * (P + 1)))
    black = AnalyseHarmonique(ouverte)
    for w in (0.4, 1.0, 3.0):
        phase, gain = black.point_de_black(sp.Float(w))
        assert proches(
            float(gain), 20 * math.log10(1 / (w * math.sqrt(1 + w * w))), 1e-6
        )
        assert proches(float(phase), -90 - math.degrees(math.atan(w)), 1e-6)
    pulsation = marges_de_stabilite(ouverte.expression).pulsation_de_phase
    assert proches(
        float(phase_ajoutee_par_un_retard(2, pulsation)), -2 * pulsation, 1e-12
    )
    marge = math.radians(marges_de_stabilite(ouverte.expression).de_phase)
    assert float(marge_de_phase_avec_retard(marge, 2, pulsation)) < 0


def test_les_marges_de_la_courbe_de_black_de_l_article() -> None:
    """Article « Les marges » : `L = 1/(p(p+1)^2)`, `MG = 2`, `MP ~ 21,8 degres`."""
    ouverte = 1 / (P * (P + 1) ** 2)
    marges = marges_de_stabilite(ouverte)
    assert proches(marges.de_gain, 2.0, 1e-3)
    w1 = float(sp.nsolve(W * (1 + W**2) - 1, W, 0.7))
    attendu = 180 - 90 - 2 * math.degrees(math.atan(w1))
    assert proches(marges.de_phase, attendu, 1e-3)
    assert proches(marges.de_gain_en_decibels(), 20 * math.log10(2), 1e-3)
    # Juin 2016 (4) : multiplier par K ne change pas les phases, divise MG par K.
    assert proches(marges_de_stabilite(3 * ouverte).de_gain, 2 / 3, 1e-3)
    assert proches(marges_de_stabilite(3 * ouverte).pulsation_de_gain, 1.0, 1e-3)
