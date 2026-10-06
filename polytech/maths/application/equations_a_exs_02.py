"""Mathematique, exercices 2, I.1 : les equations 1 a 4.

Deux a variables separees, deux homogenes de degre 0. Source :
`Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, page 4.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application.resolue_exs_02 import K, Resolue, X, Y
from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.facteur_integrant import (
    facteur_integrant,
    rendre_exacte,
)
from polytech.maths.domain.differentielles.homogenes import Z, equation_en_z
from polytech.maths.domain.differentielles.separables import solutions_constantes

SEPAREES = "variables separees"
HOMOGENE = "homogene de degre 0"


def _separee(en_x: sp.Expr, en_y: sp.Expr, solution: sp.Basic) -> Resolue:
    """`y' = f(x) g(y)` : les deux integrales, puis leurs primitives."""
    membres = sp.Eq(sp.Integral(1 / en_y, Y), sp.Integral(en_x, X))
    en_valeur_absolue = sp.integrate(1 / en_y, Y).replace(
        sp.log, lambda argument: sp.log(sp.Abs(argument))
    )
    primitives = sp.Eq(en_valeur_absolue, sp.integrate(en_x, X))
    return Resolue(en_x * en_y, SEPAREES, (membres, primitives), solution)


def _homogene(pente: sp.Expr, solution: sp.Basic) -> Resolue:
    """`z = y/x` : `f(z)`, ce que vaut `x z'`, et la primitive de `dz/(x z')`."""
    en_z, x_z_prime = equation_en_z(pente)
    inverse = sp.simplify(1 / x_z_prime).replace(
        sp.tanh, lambda argument: sp.sinh(argument) / sp.cosh(argument)
    )
    primitive = sp.simplify(sp.integrate(inverse, Z))
    return Resolue(pente, HOMOGENE, (en_z, x_z_prime, primitive), solution)


def equation_1() -> Resolue:
    """`y y' - x y^2 + x = 0`, soit `y' = x (y^2 - 1)/y`."""
    return _separee(X, (Y**2 - 1) / Y, sp.sqrt(1 + K * sp.exp(X**2)))


def constantes_de_la_1() -> tuple[sp.Expr, ...]:
    """Les solutions constantes, que la division par `y^2 - 1` ecarte."""
    return solutions_constantes((Y**2 - 1) / Y)


def equation_2() -> Resolue:
    """`x y dx + (x^2 + y^2) dy = 0`."""
    solution = sp.Eq(Y * sp.sqrt(Y**2 + 2 * X**2), K)
    return _homogene(-X * Y / (X**2 + Y**2), solution)


def second_chemin_de_la_2() -> tuple[sp.Expr, sp.Expr]:
    """Le facteur integrant de la meme equation, et le potentiel qu'il donne."""
    forme = FormeDifferentielle(X * Y, X**2 + Y**2)
    return facteur_integrant(forme), rendre_exacte(forme).potentiel()


def equation_3() -> Resolue:
    """`y e^(2x) dx - (1 + e^(2x)) dy = 0`."""
    en_x = sp.exp(2 * X) / (1 + sp.exp(2 * X))
    return _separee(en_x, Y, K * sp.sqrt(1 + sp.exp(2 * X)))


def equation_4() -> Resolue:
    """`(2x sinh(y/x) + 3y cosh(y/x)) dx - 3x cosh(y/x) dy = 0`, pour `x > 0`."""
    rapport = Y / X
    pente = (2 * X * sp.sinh(rapport) + 3 * Y * sp.cosh(rapport)) / (
        3 * X * sp.cosh(rapport)
    )
    solution = sp.Eq(sp.sinh(rapport) / X ** sp.Rational(2, 3), K)
    return _homogene(pente, solution)
