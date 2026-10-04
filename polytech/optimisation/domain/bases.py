"""Chapitre I.4.3 : bases, solutions de base, et les sommets qu'elles caracterisent.

Le probleme est sous forme standard, `T x = d`, `x >= 0`, et le cours
suppose le systeme **non redondant** : `r(T) = m`. Une **base** est une
sous-matrice `B` de `T`, carree, d'ordre `m` et **reguliere**. Les `m`
variables de ses colonnes sont les **variables de base**, les `n - m`
autres les **variables hors base**. La **solution de base** s'obtient en
annulant les variables hors base : `x_B = B^-1 d`, `x_R = 0`.

Elle est **admissible** si `x_B >= 0`, et **degeneree** si une variable
de base au moins est nulle. La propriete fondamentale : chaque sommet du
polyedre `D` correspond a une solution de base admissible, et
inversement -- mais un meme sommet peut correspondre a plusieurs bases,
et c'est exactement la degenerescence.

Toutes les sous-matrices `m x m` ne sont pas des bases : celles dont les
colonnes sont liees sont ecartees, comme `{1,3,5}` page 45 du cours.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.optimisation.domain.formes import FormeStandard


@dataclass(frozen=True, slots=True)
class SolutionDeBase:
    """La solution associee a une base : ses indices, ses valeurs, son point."""

    indices: tuple[int, ...]
    valeurs_de_base: tuple[sp.Expr, ...]
    point: sp.Matrix

    @property
    def admissible(self) -> bool:
        """Vrai si toutes les variables de base sont non negatives."""
        return all(v >= 0 for v in self.valeurs_de_base)

    @property
    def degeneree(self) -> bool:
        """Vrai si une variable de base au moins est nulle."""
        return any(v == 0 for v in self.valeurs_de_base)


def solutions_de_base(forme: FormeStandard) -> tuple[SolutionDeBase, ...]:
    """Toutes les solutions de base, une par sous-matrice reguliere d'ordre `m`."""
    m, n = forme.matrice.shape
    if forme.matrice.rank() < m:
        msg = f"systeme redondant : r(T) = {forme.matrice.rank()} pour m = {m}"
        raise ValueError(msg)
    solutions = []
    for indices in combinations(range(n), m):
        base = forme.matrice[:, list(indices)]
        if base.det() == 0:
            continue
        valeurs = tuple(base.solve(forme.seconds_membres))
        point = sp.zeros(n, 1)
        for indice, valeur in zip(indices, valeurs, strict=True):
            point[indice, 0] = valeur
        solutions.append(SolutionDeBase(indices, valeurs, point))
    return tuple(solutions)


def sommets(forme: FormeStandard, variables: int) -> tuple[sp.Matrix, ...]:
    """Les sommets de `D`, lus sur les `variables` premieres composantes.

    Un sommet degenere est rendu une seule fois, bien que plusieurs bases
    y menent.
    """
    points: list[sp.Matrix] = []
    for solution in solutions_de_base(forme):
        point = solution.point[:variables, :]
        if solution.admissible and point not in points:
            points.append(point)
    return tuple(points)
