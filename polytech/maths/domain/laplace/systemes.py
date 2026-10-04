"""Resoudre par Laplace un systeme d'equations differentielles lineaires couplees.

Quand les derivees de plusieurs inconnues se melent dans chaque equation
-- `x'' + y' + x = e^t`, `x' + y'' = 1` -- le systeme ne s'ecrit pas
`y' = A y`. La transformee de Laplace s'en accommode : chaque equation
devient une equation **algebrique** en `X(p)`, `Y(p)`, les conditions
initiales y entrant par la formule de la derivee ; on resout le systeme
lineaire, puis on revient aux originaux.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.laplace.proprietes import de_la_derivee_n
from polytech.maths.domain.symboles import IMAGE, TEMPS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _ordre(equations: Sequence[sp.Eq], inconnue: sp.Expr) -> int:
    ordres = [
        d.derivative_count
        for equation in equations
        for d in (equation.lhs - equation.rhs).atoms(sp.Derivative)
        if d.expr == inconnue
    ]
    return max(ordres, default=0)


def _en_image(
    equation: sp.Eq,
    images: Mapping[sp.Expr, sp.Symbol],
    initiales: Mapping[sp.Expr, Sequence[object]],
) -> sp.Expr:
    """L'equation transformee, egalee a zero."""
    reste = sp.expand(equation.lhs - equation.rhs)
    transformee = sp.Integer(0)
    for inconnue, image in images.items():
        valeurs = initiales[inconnue]
        for ordre in range(len(valeurs), -1, -1):
            terme = inconnue.diff(TEMPS, ordre) if ordre else inconnue
            coefficient = reste.coeff(terme)
            transformee += coefficient * de_la_derivee_n(image, valeurs[:ordre])
            reste = sp.expand(reste - coefficient * terme)
    return transformee + sp.laplace_transform(reste, TEMPS, IMAGE, noconds=True)


def resoudre_le_systeme(
    equations: Sequence[sp.Eq],
    initiales: Mapping[sp.Expr, Sequence[object]],
) -> dict[sp.Expr, sp.Expr]:
    """Les originaux `x(t)`, `y(t)`, ... pour `t >= 0`."""
    for inconnue, valeurs in initiales.items():
        ordre = _ordre(equations, inconnue)
        if len(valeurs) != ordre:
            msg = (
                f"{len(valeurs)} conditions initiales pour {inconnue}, "
                f"qui est derivee {ordre} fois"
            )
            raise ValueError(msg)
    images = {
        inconnue: sp.Symbol(f"L_{rang}") for rang, inconnue in enumerate(initiales)
    }
    algebriques = [_en_image(e, images, initiales) for e in equations]
    resolues = sp.solve(algebriques, list(images.values()), dict=True)[0]
    originaux = {}
    for inconnue, image in images.items():
        inverse = sp.inverse_laplace_transform(
            sp.simplify(resolues[image]), IMAGE, TEMPS, noconds=True
        )
        originaux[inconnue] = sp.simplify(inverse.subs(sp.Heaviside(TEMPS), 1))
    return originaux
