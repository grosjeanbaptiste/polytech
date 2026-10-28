"""Fiche d'electronique : les courbes des figures TikZ, refaites par la lib.

Seules les figures dont l'expression pgfplots porte un nombre calculable
sont reprises ici ; les autres sont des schemas.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.domain.aop_interne import bande_en_boucle_ouverte
from polytech.electronique.domain.butterworth import poles_de_butterworth
from polytech.electronique.domain.capteur import sensibilite
from polytech.electronique.domain.reponse_du_biquad import module_carre
from polytech.maths.domain.comparaison import proches
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.trace_asymptotique import trace_d_une_transmittance

M = sp.Symbol("m", positive=True)


def test_les_asymptotes_de_la_figure_du_bode_sont_celles_de_la_transmittance() -> None:
    """Article « Bode asymptotique » : `10 (1 + s/10) / ((1 + s/100)(1 + s/1000))`."""
    trace = trace_d_une_transmittance(
        10 * (1 + P / 10) / ((1 + P / 100) * (1 + P / 1000))
    )
    assert [c for c, _ in trace.cassures] == [10.0, 100.0, 1000.0]
    for x in (1.0, 30.0, 400.0, 5e4):
        reel = 10 * max(1, x / 10) / (max(1, x / 100) * max(1, x / 1000))
        assert proches(trace.gain_en_decibels(x), 20 * math.log10(reel), 1e-9)
    assert proches(trace.gain_en_decibels(1.0), 20.0, 1e-9)


def test_la_figure_du_produit_gain_bande_garde_un_million() -> None:
    """Article « Le produit gain-bande » : `1e5` a `10 Hz`, `20` a `5e4 Hz`."""
    assert bande_en_boucle_ouverte(1e6, 1e5) == 10.0
    assert 1e5 * bande_en_boucle_ouverte(1e6, 1e5) == 20 * 5e4


def test_les_poles_de_la_figure_de_butterworth_sont_a_112_5_degres_et_plus() -> None:
    """Article « Reconnaitre un Butterworth » : angles `90 + 180 (2k-1)/8`."""
    poles = poles_de_butterworth(4)
    figure = [90 + 180 * (2 * k - 1) / 8 for k in range(1, 5)]
    angles = sorted(math.degrees(math.atan2(p.imag, p.real)) % 360 for p in poles)
    assert all(proches(a, f, 1e-9) for a, f in zip(angles, figure, strict=True))


def test_les_trois_courbes_de_la_section_du_second_ordre() -> None:
    """Article « La section passe-bas » : `Q = 2`, `1/sqrt 2`, `1/2`."""
    for qualite, terme in ((2.0, 1 / 4), (1 / math.sqrt(2), 2.0), (0.5, 4.0)):
        for x in (0.1, 0.8, 1.0, 3.0, 10.0):
            figure = -10 * math.log10((1 - x**2) ** 2 + terme * x**2)
            lib = 10 * math.log10(module_carre(x, qualite))
            assert proches(figure, lib, 1e-9)


def test_la_tangente_de_la_figure_du_capteur_est_la_pente_en_25() -> None:
    """Article « La sensibilite » : `s = 1 - e^{-m/30}`, pente `e^{-25/30}/30`."""
    pente = float(sensibilite(1 - sp.exp(-M / 30), M, 25))
    assert proches(pente, math.exp(-25 / 30) / 30, 1e-12)
