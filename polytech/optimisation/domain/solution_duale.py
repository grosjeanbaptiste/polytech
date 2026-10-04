"""Chapitre IV.4 : la solution duale associee a une base.

A une base `B` du primal `min c x`, `T x = d`, `x >= 0`, le cours associe
deux solutions :

* la solution de base **primale** `x_B = B^-1 d`, `x_R = 0` ;
* la solution **duale** `u(B) = c_B B^-1`, intersection des `m`
  contraintes duales `u t_i = c_i`, `i` dans la base.

Et deux conditions, qui font ensemble l'optimalite :

1. **primale admissible** : `x_B >= 0` ;
2. **duale admissible** : `a_0j = u(B) t_j - c_j <= 0` pour tout `j`,
   c'est-a-dire `u(B) T <= c` -- `u(B)` est admissible pour le dual.

L'algorithme simplexe garde la premiere et cherche la seconde ;
l'algorithme dual (chapitre V) garde la seconde et cherche la premiere.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.optimisation.domain.formes import FormeStandard


@dataclass(frozen=True, slots=True)
class BaseDuCouple:
    """Une base, ses deux solutions, et ce que chacune a d'admissible."""

    indices: tuple[int, ...]
    x: tuple[sp.Expr, ...]
    u: tuple[sp.Expr, ...]
    primale_admissible: bool
    duale_admissible: bool

    @property
    def optimale(self) -> bool:
        """Vrai si la base est a la fois primale et duale admissible."""
        return self.primale_admissible and self.duale_admissible


def _matrice_de_base(forme: FormeStandard, indices: tuple[int, ...]) -> sp.Matrix:
    base = forme.matrice[:, [i - 1 for i in indices]]
    if base.det() == 0:
        msg = f"{indices} n'est pas une base : sous-matrice non reguliere"
        raise ValueError(msg)
    return base


def solution_duale(
    forme: FormeStandard, indices: tuple[int, ...]
) -> tuple[sp.Expr, ...]:
    """`u(B) = c_B B^-1`."""
    base = _matrice_de_base(forme, indices)
    return tuple(forme.couts[:, [i - 1 for i in indices]] * base.inv())


def base_du_couple(forme: FormeStandard, indices: tuple[int, ...]) -> BaseDuCouple:
    """Les solutions primale et duale de la base, et leur admissibilite."""
    valeurs = _matrice_de_base(forme, indices).solve(forme.seconds_membres)
    x = [valeurs[indices.index(j), 0] if j in indices else 0 for j in _tous(forme)]
    u = solution_duale(forme, indices)
    couts_reduits = sp.Matrix([u]) * forme.matrice - forme.couts
    return BaseDuCouple(
        indices=indices,
        x=tuple(x),
        u=u,
        primale_admissible=all(v >= 0 for v in valeurs),
        duale_admissible=all(v <= 0 for v in couts_reduits),
    )


def _tous(forme: FormeStandard) -> range:
    return range(1, forme.matrice.cols + 1)


def bases_du_couple(forme: FormeStandard) -> tuple[BaseDuCouple, ...]:
    """Toutes les bases, dans l'ordre lexicographique des indices (page 110)."""
    return tuple(
        base_du_couple(forme, indices)
        for indices in combinations(_tous(forme), forme.matrice.rows)
        if forme.matrice[:, [i - 1 for i in indices]].det() != 0
    )
