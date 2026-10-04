"""Chapitre I.3.1, I.3.4 et I.4 : combinaisons convexes et positives.

`somme(lambda_k x_k)` est une **combinaison convexe** des `x_k` si
`somme(lambda_k) = 1` et `lambda_k >= 0`. Un ensemble convexe contient
toutes les combinaisons convexes de ses points, et tous les points d'un
polytope sont combinaison convexe de ses sommets.

Decider si un point en est une revient a chercher des `lambda`
admissibles : un programme lineaire sous forme standard, dont les
variables sont les `lambda`. S'il a une solution, il a une solution de
base admissible -- et il suffit de les passer en revue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.bases import (
    solutions_de_base,
    supprimer_les_equations_redondantes,
)
from polytech.optimisation.domain.formes import FormeStandard

if TYPE_CHECKING:
    from collections.abc import Sequence


def _a_une_solution_positive(matrice: sp.Matrix, second: sp.Matrix) -> bool:
    """Vrai s'il existe `lambda >= 0` tel que `matrice lambda = second`."""
    if matrice.row_join(second).rank() > matrice.rank():
        # Systeme impossible : aucun `lambda`, meme de signe quelconque.
        return False
    systeme = FormeStandard(sp.zeros(1, matrice.cols), matrice, second)
    reduit = supprimer_les_equations_redondantes(systeme)
    return any(solution.admissible for solution in solutions_de_base(reduit))


def est_combinaison_convexe(
    point: Sequence[object],
    points: Sequence[Sequence[object]],
) -> bool:
    """Vrai si `point = somme(lambda_k points_k)`, `lambda >= 0`, `somme = 1`."""
    colonnes = sp.Matrix([[*p, 1] for p in points]).T
    return _a_une_solution_positive(colonnes, sp.Matrix([*point, 1]))


def est_combinaison_positive(
    vecteur: sp.Matrix,
    generateurs: Sequence[sp.Matrix],
) -> bool:
    """Vrai si `vecteur` est dans le cone polyedrique engendre par `generateurs`.

    Page 37 : `d` dans le cone des `t_j` est la condition necessaire et
    suffisante pour qu'il existe des solutions admissibles.
    """
    return _a_une_solution_positive(sp.Matrix.hstack(*generateurs), vecteur)
