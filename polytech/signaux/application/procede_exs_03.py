"""Feuille « Systemes », exercice 4 : un procede lineaire a quatre variables.

Les equations sont deja lineaires : le regime s'obtient en annulant les
derivees, et le schema-bloc en transformant chaque equation.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.transmittance import Transmittance

COMMANDE: Final = sp.Integer(5)
_DEMI: Final = sp.Rational(1, 2)


def regime() -> dict[str, sp.Expr]:
    """Les cinq valeurs de repos quand `u` est constant et egal a 5."""
    x1, x2, x3, x4, y = sp.symbols("x1 x2 x3 x4 y")
    equations = (
        x1 - COMMANDE,
        -_DEMI * x2 + _DEMI * COMMANDE,
        -2 * x3 + 10 * (x1 + x2),
        -x4 + 2 * COMMANDE,
        y - x4 - x3,
    )
    solution = sp.solve(equations, (x1, x2, x3, x4, y))
    return {str(symbole): valeur for symbole, valeur in solution.items()}


def blocs() -> tuple[Transmittance, Transmittance, Transmittance]:
    """`X2/U`, `X3/(X1 + X2)` et `X4/U` : un premier ordre par equation."""
    return (
        Transmittance(expression=_DEMI / (IMAGE + _DEMI)),
        Transmittance(expression=10 / (IMAGE + 2)),
        Transmittance(expression=2 / (IMAGE + 1)),
    )


def transmittance() -> Transmittance:
    """`Y/U`, en suivant le schema-bloc : deux voies qui s'ajoutent."""
    vers_x2, vers_x3, vers_x4 = blocs()
    haute = (1 + vers_x2.expression) * vers_x3.expression
    return Transmittance(expression=sp.cancel(haute + vers_x4.expression))
