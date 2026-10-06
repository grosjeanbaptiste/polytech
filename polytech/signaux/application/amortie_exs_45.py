"""Seance « Chap. 4-5 », exercices 1 et 2 : lire une sinusoide amortie.

`e^(-sigma t) cos(omega t + phi)` se dessine avec trois reperes : sa
periode, sa constante de temps `1/sigma`, sa valeur de depart. Et un
signal donne par sa transformee se dessine sans l'inverser, par ses
poles et les deux theoremes des valeurs extremes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

SIGMA: Final = 10.0
PULSATION: Final = 2 * math.pi * 100
PHASE: Final = math.pi / 4
IMAGE_DE_LA_TENSION: Final = P / (P**2 + 2 * P + 2)
"""`U_C(p)` de l'exercice 2."""
CONSTANTES_POUR_S_ETEINDRE: Final = 5
"""Le nombre de constantes de temps au bout desquelles on dit le signal amorti."""


@dataclass(frozen=True, slots=True)
class Reperes:
    """Ce qu'il faut porter sur le dessin de l'exercice 1."""

    periode: float
    constante: float
    depart: float
    periodes_par_constante: float


def reperes_de_la_sinusoide() -> Reperes:
    """La periode, la constante de temps, la valeur en zero, et leur rapport."""
    periode, constante = 2 * math.pi / PULSATION, 1 / SIGMA
    return Reperes(periode, constante, math.cos(PHASE), constante / periode)


def valeurs_extremes() -> tuple[sp.Expr, sp.Expr]:
    """`u(0+)` et `u(infini)` : les limites de `p U(p)` a l'infini et en zero."""
    produit = sp.cancel(P * IMAGE_DE_LA_TENSION)
    return sp.limit(produit, P, sp.oo), sp.limit(produit, P, 0)


def amortissement_et_pulsation() -> tuple[sp.Expr, sp.Expr]:
    """`sigma` et `omega` des poles `-sigma +- j omega`."""
    denominateur = sp.fraction(IMAGE_DE_LA_TENSION)[1]
    pole = max(sp.roots(sp.Poly(denominateur, P)), key=sp.im)
    return -sp.re(pole), sp.im(pole)


def duree_et_periode() -> tuple[sp.Expr, sp.Expr]:
    """Le temps au bout duquel le signal est amorti, et sa periode propre."""
    sigma, pulsation = amortissement_et_pulsation()
    return CONSTANTES_POUR_S_ETEINDRE / sigma, 2 * sp.pi / pulsation


def original() -> sp.Expr:
    """`u_C(t)`, par l'inversion formelle que l'enonce demande d'eviter."""
    brut = sp.inverse_laplace_transform(IMAGE_DE_LA_TENSION, P, T)
    return sp.simplify(brut.subs(sp.Heaviside(T), 1))
