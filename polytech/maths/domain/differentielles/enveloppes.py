"""L'enveloppe d'une famille de courbes `G(x, y, c) = 0`.

C'est une courbe tangente, en chacun de ses points, a une courbe de la
famille. Son point de parametre `c` est sur la courbe `C_c`, et son
vecteur tangent y est perpendiculaire au gradient de `G` : il reste

    G(x, y, c) = 0,     dG/dc (x, y, c) = 0.

Si la famille est la solution generale d'une equation differentielle,
l'enveloppe en est aussi solution -- **singuliere** quand elle n'appartient
pas a la famille.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE

type Point = tuple[sp.Expr, sp.Expr]


def enveloppe(famille: sp.Expr, parametre: sp.Symbol) -> tuple[Point, ...]:
    """Les branches `(x(c), y(c))` de l'enveloppe ; leve s'il n'y en a pas."""
    systeme = [sp.Eq(famille, 0), sp.Eq(sp.diff(famille, parametre), 0)]
    branches = sp.solve(systeme, [VARIABLE, INCONNUE], dict=True)
    if not branches:
        msg = "aucune enveloppe : le systeme G = 0, dG/dc = 0 n'a pas de solution"
        raise ValueError(msg)
    return tuple(
        (sp.simplify(branche[VARIABLE]), sp.simplify(branche[INCONNUE]))
        for branche in branches
    )
