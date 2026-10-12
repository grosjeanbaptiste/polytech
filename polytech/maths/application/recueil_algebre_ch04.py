"""Recueil d'algebre, chapitre IV : division, racines, factorisation, Viete.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 6 de
la section IV.4 (pages 52 et 53).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import sympy as sp

from polytech.maths.application.formatage import coordonnees
from polytech.maths.application.recueil_algebre_ch01_racines import racines_exactes
from polytech.maths.domain.algebre.polynomes_exacts import division
from polytech.maths.domain.algebre.racines_exactes import (
    factoriser_dans_r,
    valeur_symetrique_des_racines,
)
from polytech.maths.domain.symboles import VARIABLE as X

DIVISIONS: Final = {
    "a": (X**3 - 3 * X + 1, X - 1),
    "b": (3 * X**5 - 4 * X**3 + X + 5, X**3 - 2 * X + 7),
}
POLYNOMES_2: Final = {
    "a": X**4 - 5 * X**2 + 4,
    "b": 3 * X**3 - 19 * X**2 + 34 * X - 14,
    "c": 6 * X**4 - 5 * X**3 - 38 * X**2 - 5 * X + 6,
}
DIVISEURS_2: Final = {"b": X - sp.Rational(7, 3), "c": (X + 2) * (X - 3)}
"""Ce par quoi on divise, une fois les racines rationnelles trouvees."""
POLYNOMES_3: Final = {
    "a": X**4 + 4 * X**2 + 3,
    "b": X**4 + X**3 - 11 * X**2 + X - 12,
    "c": 6 * X**4 - 13 * X**3 + 13 * X - 6,
    "d": X**4 + X**2 + 1,
}
POLYNOME_4: Final = X**6 + X**5 + 2 * X**3 - 3 * X**2 + X - 2
DEGRE: Final = sp.Symbol("n", integer=True, positive=True)
POLYNOME_6: Final = X**3 + 3 * X - 2
RACINES_6: Final = sp.symbols("x1:4")
INTERVALLE: Final = (-2.4, 2.4)
POINTS: Final = 49


def exercice_1() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """Quotient et reste des deux divisions."""
    return {lettre: division(p, q) for lettre, (p, q) in DIVISIONS.items()}


def exercice_2() -> dict[str, tuple[sp.Expr, ...]]:
    """Toutes les racines des trois polynomes."""
    return {lettre: racines_exactes(p) for lettre, p in POLYNOMES_2.items()}


def quotients_du_2() -> dict[str, sp.Expr]:
    """Ce qui reste a resoudre apres les racines rationnelles : un second degre."""
    return {k: division(POLYNOMES_2[k], d)[0] for k, d in DIVISEURS_2.items()}


def exercice_3() -> dict[str, tuple[sp.Expr, tuple[sp.Expr, ...]]]:
    """Le coefficient dominant et les facteurs reels de chaque polynome."""
    return {lettre: factoriser_dans_r(p) for lettre, p in POLYNOMES_3.items()}


def exercice_4() -> tuple[sp.Expr, dict[sp.Expr, int]]:
    """Le quotient par `(x^2 + 1)^2`, et toutes les racines avec leur ordre."""
    quotient, _ = division(POLYNOME_4, (X**2 + 1) ** 2)
    racines = {sp.I: 2, -sp.I: 2}
    racines |= dict.fromkeys(racines_exactes(quotient), 1)
    return quotient, racines


def exercice_5() -> tuple[sp.Expr, sp.Expr]:
    """`a` et `b` tels que `a z^n + b z + 1` et sa derivee s'annulent en 1."""
    a, b = sp.symbols("a b")
    polynome = a * X**DEGRE + b * X + 1
    conditions = [polynome.subs(X, 1), sp.diff(polynome, X).subs(X, 1)]
    (solution,) = sp.solve(conditions, [a, b], dict=True)
    return sp.simplify(solution[a]), sp.simplify(solution[b])


def exercice_6() -> dict[str, sp.Expr]:
    """Trois expressions symetriques des racines, sans calculer les racines."""
    x1, x2, x3 = RACINES_6
    expressions = {
        "a": x1**2 + x2**2 + x3**2,
        "b": x1**2 * x2**2 + x1**2 * x3**2 + x2**2 * x3**2,
        "c": 1 / x1 + 1 / x2 + 1 / x3,
    }
    coefficients = sp.Poly(POLYNOME_6, X).all_coeffs()
    return {
        lettre: valeur_symetrique_des_racines(coefficients, e, RACINES_6)
        for lettre, e in expressions.items()
    }


def courbe_du_2a() -> str:
    """Le graphe de `x^4 - 5 x^2 + 4`, dont les quatre racines se lisent."""
    abscisses = np.linspace(*INTERVALLE, POINTS)
    ordonnees = sp.lambdify(X, POLYNOMES_2["a"], "numpy")(abscisses)
    return coordonnees(abscisses, ordonnees, 3)
