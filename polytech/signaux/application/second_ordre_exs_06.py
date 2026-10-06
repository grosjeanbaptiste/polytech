"""Feuille « Analyse harmonique », exercice 6 : un second ordre resonant.

`10/(p^2 + 0,8 p + 1)` est peu amorti : sa courbe de gain depasse son
palier avant de tomber, et c'est ce que le trace asymptotique ne montre
pas.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.signaux.application.bode_exs_06 import argument_reel, gain_reel
from polytech.signaux.domain.bode import pulsation_de_resonance
from polytech.signaux.domain.symboles import IMAGE as P

TRANSMITTANCE: Final = 10 / (P**2 + sp.Rational(4, 5) * P + 1)


def parametres(
    transmittance: sp.Expr = TRANSMITTANCE,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`(K, w_n, xi)`, par identification a `K w_n^2/(p^2 + 2 xi w_n p + w_n^2)`."""
    numerateur, denominateur = sp.fraction(sp.cancel(transmittance))
    deux, un, zero = sp.Poly(denominateur, P).all_coeffs()
    propre = sp.sqrt(zero / deux)
    return numerateur / zero, propre, un / (2 * deux * propre)


def resonance(transmittance: sp.Expr = TRANSMITTANCE) -> tuple[float, float]:
    """La pulsation de resonance, et le depassement du palier en decibels."""
    _, propre, amortissement = (float(v) for v in parametres(transmittance))
    pulsation = pulsation_de_resonance(1 / propre, amortissement)
    facteur = 1 / (2 * amortissement * math.sqrt(1 - amortissement**2))
    return pulsation, 20 * math.log10(facteur)


def point_propre() -> tuple[float, float]:
    """Le gain, en decibels, et l'argument, en degres, a la pulsation propre."""
    _, propre, _ = parametres()
    pulsation = float(propre)
    return gain_reel(TRANSMITTANCE, pulsation), argument_reel(TRANSMITTANCE, pulsation)
