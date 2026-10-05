"""Controler une primitive annoncee, sans la recalculer.

Deriver la reponse et la comparer a l'enonce est le seul controle qui ne
depende pas de la forme sous laquelle la primitive est ecrite : deux
primitives justes peuvent ne se ressembler en rien. Quand des radicaux ou
des valeurs absolues empechent la simplification formelle, la comparaison
se fait en des points du domaine, a grande precision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

CHIFFRES = 30
"""La precision de l'evaluation numerique."""


def ecart_de_derivation(
    primitive: sp.Expr, integrande: sp.Expr, points: Sequence[object]
) -> sp.Expr:
    """Le plus grand `|F'(x) - f(x)|` sur les points donnes ; leve sans point."""
    if not points:
        msg = "aucun point de controle : la primitive ne serait pas verifiee"
        raise ValueError(msg)
    ecart = sp.diff(primitive, VARIABLE) - integrande
    return max(sp.Abs(sp.N(ecart.subs(VARIABLE, point), CHIFFRES)) for point in points)
