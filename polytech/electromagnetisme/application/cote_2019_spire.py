"""Exercice cote 2019, question 2 : une spire carree dans une induction variable.

L'induction est uniforme, inclinee de `alpha` sur le plan de la spire, et
sa grandeur decrit deux triangles. Seule sa composante normale a la spire
compte pour le flux.
"""

from __future__ import annotations

import itertools
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.induction import (
    force_electromotrice,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

T: Final = TEMPS
COTE: Final = sp.Symbol("a", positive=True)
ALPHA: Final = sp.Symbol("alpha", positive=True)
SOMMETS: Final = (
    (sp.Integer(0), sp.Integer(0)),
    (sp.Rational(1, 2), sp.Integer(1)),
    (sp.Integer(1), sp.Integer(0)),
    (sp.Rational(3, 2), sp.Integer(1)),
    (sp.Integer(2), sp.Integer(0)),
)
"""Les sommets `(t, B)` du graphe de l'enonce, en secondes et en teslas."""


def flux_et_fem(induction: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Le flux a travers la spire orientee de 1 vers 2, et `E12 = -d phi / dt`.

    Parcourue de 1 vers 2, la spire a pour normale `-u_z` : le flux de la
    composante `B sin(alpha)` y est negatif.
    """
    flux = -(COTE**2) * sp.sin(ALPHA) * induction
    return flux, force_electromotrice(flux)


def induction(instant: object) -> sp.Expr:
    """`B(t)`, lue sur les segments du graphe ; nulle apres deux secondes."""
    t = sp.sympify(instant)
    for (debut, depart), (fin, arrivee) in itertools.pairwise(SOMMETS):
        if debut <= t <= fin:
            return depart + (arrivee - depart) * (t - debut) / (fin - debut)
    return sp.Integer(0)


def paliers() -> tuple[tuple[sp.Expr, sp.Expr, sp.Expr], ...]:
    """Pour chaque segment : son debut, sa fin, et la valeur constante de `E12`."""
    resultat = []
    for (debut, depart), (fin, arrivee) in itertools.pairwise(SOMMETS):
        segment = depart + (arrivee - depart) * (T - debut) / (fin - debut)
        resultat.append((debut, fin, flux_et_fem(segment)[1]))
    resultat.append((SOMMETS[-1][0], sp.oo, flux_et_fem(sp.Integer(0))[1]))
    return tuple(resultat)
