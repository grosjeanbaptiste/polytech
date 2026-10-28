"""Lire une expression de pulsation : son exposant, son module en un point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Mapping


def exposant_de(expression: sp.Expr, variable: sp.Expr) -> int:
    """`n` de `k variable^n` ; leve si l'expression n'est pas un monome."""
    logarithmique = sp.simplify(variable * sp.diff(expression, variable) / expression)
    if not logarithmique.is_Integer:
        msg = f"pas un monome en {variable} : {expression}"
        raise ValueError(msg)
    return int(logarithmique)


def module_de(expression: sp.Expr, valeurs: Mapping[sp.Expr, complex]) -> float:
    """`|expression|` une fois chaque symbole remplace par sa valeur."""
    return abs(complex(expression.subs(valeurs)))
