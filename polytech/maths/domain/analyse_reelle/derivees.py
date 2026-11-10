"""Nombres dérivés à gauche et à droite, accroissements finis.

Syllabus de calcul différentiel et intégral 1, sections 5.1.2, 5.1.3 et
5.2.2. Le nombre dérivé en `a` est la limite du taux d'accroissement
`(f(x) - f(a)) / (x - a)` ; on le prend de chaque côté, et il peut valoir
`+oo` ou `-oo` (nombres dérivés infinis). `f` est dérivable en `a` quand
les deux nombres existent, sont finis et coïncident.

Accroissements finis : si `f` est continue sur `[a, b]` et dérivable sur
`]a, b[`, il existe `c` dans `]a, b[` avec `f'(c) = (f(b) - f(a)) / (b - a)` ;
quand `f(a) = f(b)`, c'est le théorème de Rolle (`f'(c) = 0`).
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.analyse_reelle.continuite import (
    limite_a_droite,
    limite_a_gauche,
)
from polytech.maths.domain.symboles import VARIABLE as X


def nombres_derives(expression: sp.Expr, point: object) -> tuple[sp.Expr, sp.Expr]:
    """Les nombres dérivés à gauche et à droite de `f` en `point`."""
    taux = (expression - expression.subs(X, point)) / (X - sp.sympify(point))
    return limite_a_gauche(taux, point), limite_a_droite(taux, point)


def est_derivable_en(expression: sp.Expr, point: object) -> bool:
    """Vrai si les deux nombres dérivés existent, sont finis et égaux."""
    gauche, droite = nombres_derives(expression, point)
    return gauche == droite and bool(gauche.is_finite)


def points_des_accroissements_finis(
    expression: sp.Expr, a: object, b: object
) -> tuple[sp.Expr, ...]:
    """Les `c` de `]a, b[` ou `f'` vaut la pente de la corde ; leve si `b <= a`."""
    borne_a, borne_b = sp.sympify(a), sp.sympify(b)
    if not borne_a < borne_b:
        msg = f"il faut a < b, recu a = {a}, b = {b}"
        raise ValueError(msg)
    pente = (expression.subs(X, borne_b) - expression.subs(X, borne_a)) / (
        borne_b - borne_a
    )
    solutions = sp.solveset(
        sp.diff(expression, X) - pente, X, sp.Interval.open(borne_a, borne_b)
    )
    return tuple(sorted(solutions, key=float))
