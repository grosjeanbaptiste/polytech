"""Les figures des corriges d'examen, et les aires que la quadrature recoupe.

Une integrale definie est une aire : le corrige la dessine, et la mesure une
seconde fois par quadrature numerique -- sans primitive, donc sans rien
devoir au calcul qu'elle controle.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import sympy as sp
from scipy import integrate

from polytech.maths.application import examen_2021_analyse as an
from polytech.maths.application.formatage import coordonnees
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as T
from polytech.maths.domain.symboles import VARIABLE as X

POINTS: Final = 41
DECIMALES: Final = 4
FIN_DE_LA_SOLUTION_8: Final = 1.5
"""Juste avant `pi / 2`, ou `tan t` cesse d'etre definie."""

INTEGRANDE_2022: Final = sp.log(1 + X) / X**2
INTEGRANDE_2025: Final = 4 * X**2 / (X**4 - 1)


def _trace(expression: sp.Expr, variable: sp.Symbol, debut: float, fin: float) -> str:
    abscisses = np.linspace(debut, fin, POINTS)
    ordonnees = sp.lambdify(variable, expression, "numpy")(abscisses)
    return coordonnees(abscisses, np.asarray(ordonnees, dtype=float), DECIMALES)


def _aire(expression: sp.Expr, debut: float, fin: float) -> float:
    valeur, _ = integrate.quad(sp.lambdify(X, expression, "math"), debut, fin)
    return float(valeur)


def integrande_2022() -> str:
    """`ln(1 + x) / x^2` entre 1 et 2."""
    return _trace(INTEGRANDE_2022, X, 1.0, 2.0)


def integrande_2025() -> str:
    """`4 x^2 / (x^4 - 1)` entre 2 et 3."""
    return _trace(INTEGRANDE_2025, X, 2.0, 3.0)


def aire_2022() -> float:
    """L'integrale de janvier 2022, par quadrature."""
    return _aire(INTEGRANDE_2022, 1.0, 2.0)


def aire_2025() -> float:
    """L'integrale de janvier 2025, par quadrature."""
    return _aire(INTEGRANDE_2025, 2.0, 3.0)


def solution_8_de_2021() -> str:
    """La solution de l'equation (8) de janvier 2021, depuis `y(0) = 6`."""
    return _trace(an.solution_de(8), T, 0.0, FIN_DE_LA_SOLUTION_8)
