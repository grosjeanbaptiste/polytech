"""Resoudre par echelonnement, puis en remontant (syllabus d'algebre, III.1).

On echelonne la matrice completee `(A | b)` ; le systeme triangulaire
obtenu se resout « en commencant par la derniere equation » : chaque
equation donne la variable de son pivot en fonction de celles qui la
suivent. Les variables sans pivot sont libres et recoivent un parametre.
Une ligne sans pivot dont le second membre n'est pas nul est une equation
`0 = c` : le systeme est impossible. C'est l'exemple 2 de la page 35,
la ou `par_echelonnement_simple` poursuit jusqu'a l'echelonnement simple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.echelonnement import (
    colonnes_des_pivots,
    echelonner,
)
from polytech.maths.domain.algebre.solutions import PARAMETRES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.maths.domain.algebre.systemes import SystemeLineaire


def par_remontee(
    systeme: SystemeLineaire, parametres: Sequence[sp.Symbol] = PARAMETRES
) -> tuple[sp.Expr, ...]:
    """Les inconnues, de la derniere a la premiere, les libres en parametres."""
    n = int(systeme.matrice.cols)
    completee = systeme.matrice.row_join(systeme.second_membre)
    triangle = echelonner(completee, colonnes=n).matrice
    pivots = colonnes_des_pivots(triangle[:, :n])
    for ligne, pivot in enumerate(pivots):
        if pivot is None and triangle[ligne, n] != 0:
            reste = triangle[ligne, n]
            msg = f"systeme impossible : la ligne {ligne + 1} dit 0 = {reste}"
            raise ValueError(msg)
    libres = [j for j in range(n) if j not in pivots]
    if len(libres) > len(parametres):
        msg = f"{len(libres)} variables libres pour {len(parametres)} parametres"
        raise ValueError(msg)
    valeurs: dict[int, sp.Expr] = dict(zip(libres, parametres, strict=False))
    for ligne in reversed(range(len(pivots))):
        pivot = pivots[ligne]
        if pivot is None:
            continue
        reste = sum(triangle[ligne, j] * valeurs[j] for j in range(pivot + 1, n))
        valeurs[pivot] = sp.simplify(
            (triangle[ligne, n] - reste) / triangle[ligne, pivot]
        )
    return tuple(valeurs[j] for j in range(n))
