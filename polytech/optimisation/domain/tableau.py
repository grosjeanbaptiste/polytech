"""Chapitre II.1 : la forme equivalente et le tableau simplexe du cours.

Enumerer les solutions de base est exclu : il y en a jusqu'a `C(n, m)`.
Le simplexe passe de l'une a l'autre, et son outil est le **tableau**.

Une base `B` etant choisie, le probleme se reecrit sous une **forme
equivalente**, en diagonalisant les contraintes sur les variables de base
et en les eliminant de la fonction economique :

    z = a_00 - somme(a_0j x_j),   x_i + somme(a_ij x_j) = a_i0

avec `a_i0 = B^-1 d`, `A = B^-1 R`, `a_0 = c_B B^-1 R - c_R` et
`a_00 = c_B B^-1 d`. Ces coefficients forment le tableau, a `m + 1`
lignes et `n - m + 1` colonnes : sa premiere colonne donne la valeur de
`z` et celle de chaque variable de base.

Les indices sont ceux du cours, a partir de 1, et l'ordre des variables
hors base est libre -- le cours laisse la variable sortie a la place de
la variable entree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.optimisation.domain.formes import FormeStandard


def nombre_maximal_de_bases(variables: int, equations: int) -> int:
    """`C(n, m)` : atteint si toutes les sous-matrices `m x m` sont regulieres."""
    return math.comb(variables, equations)


@dataclass(frozen=True, slots=True)
class TableauSimplexe:
    """Les indices de base, les indices hors base, et les coefficients `a_ij`."""

    base: tuple[int, ...]
    hors_base: tuple[int, ...]
    coefficients: sp.Matrix

    @property
    def valeur(self) -> sp.Expr:
        """`a_00` : la valeur actuelle de la fonction economique."""
        return self.coefficients[0, 0]

    def solution(self, variables: int) -> sp.Matrix:
        """La solution de base : `x_i = a_i0` en base, zero hors base."""
        point = sp.zeros(variables, 1)
        for ligne, indice in enumerate(self.base, start=1):
            point[indice - 1, 0] = self.coefficients[ligne, 0]
        return point

    def complet(self, variables: int) -> sp.Matrix:
        """Le tableau complet : `m + 1` lignes et `n + 1` colonnes."""
        complet = sp.zeros(len(self.base) + 1, variables + 1)
        complet[:, 0] = self.coefficients[:, 0]
        for rang, indice in enumerate(self.hors_base, start=1):
            complet[:, indice] = self.coefficients[:, rang]
        for ligne, indice in enumerate(self.base, start=1):
            complet[ligne, indice] = 1
        return complet


def tableau_de_la_base(
    forme: FormeStandard,
    base: tuple[int, ...],
    hors_base: tuple[int, ...] | None = None,
) -> TableauSimplexe:
    """Le tableau simplexe associe a cette base, par `B^-1`."""
    m, n = forme.matrice.shape
    if len(base) != m:
        msg = f"{len(base)} indices de base pour {m} equations"
        raise ValueError(msg)
    dehors = hors_base or tuple(j for j in range(1, n + 1) if j not in base)
    matrice_de_base = forme.matrice[:, [i - 1 for i in base]]
    if matrice_de_base.det() == 0:
        msg = f"{base} n'est pas une base : sous-matrice non reguliere"
        raise ValueError(msg)
    inverse = matrice_de_base.inv()
    reste = forme.matrice[:, [j - 1 for j in dehors]]
    couts_de_base = forme.couts[:, [i - 1 for i in base]]
    couts_hors_base = forme.couts[:, [j - 1 for j in dehors]]
    premiere = (couts_de_base * inverse * forme.seconds_membres).row_join(
        couts_de_base * inverse * reste - couts_hors_base
    )
    corps = (inverse * forme.seconds_membres).row_join(inverse * reste)
    return TableauSimplexe(tuple(base), dehors, premiere.col_join(corps))
