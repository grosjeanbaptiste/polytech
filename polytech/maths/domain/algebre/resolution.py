"""La resolution d'un systeme lineaire par echelonnement simple.

La demarche du syllabus d'algebre (III.1) : echelonner simplement la
matrice completee `(A | b)`. Dans le systeme obtenu, chaque equation isole
la variable de son pivot ; « les variables ne correspondant a aucun pivot
peuvent prendre des valeurs arbitraires, les autres s'expriment en fonction
de ces valeurs ». Un pivot dans la colonne du second membre est une
equation `0 = 1` : le systeme est impossible.

L'ensemble des solutions s'ecrit alors sous la « forme generale » : une
solution particuliere -- les variables libres a zero -- plus une
combinaison de directions, une par variable libre.

Quand la matrice porte un parametre, l'echelonnement traite comme non nul
tout pivot qui ne l'est pas identiquement : le resultat est le cas
general, et les valeurs qui annulent un pivot restent a discuter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.echelonnement import (
    colonnes_des_pivots,
    echelonner_simplement,
)
from polytech.maths.domain.algebre.solutions import PARAMETRES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.maths.domain.algebre.systemes import SystemeLineaire


@dataclass(frozen=True, slots=True)
class EnsembleDesSolutions:
    """Une solution particuliere, et une direction par variable libre."""

    particuliere: tuple[sp.Expr, ...]
    directions: tuple[tuple[sp.Expr, ...], ...]

    def forme_generale(
        self, parametres: Sequence[sp.Symbol] = PARAMETRES
    ) -> tuple[sp.Expr, ...]:
        """`particuliere + alpha d_1 + beta d_2 + ...`."""
        besoin, offre = len(self.directions), len(parametres)
        if besoin > offre:
            msg = f"trop peu de parametres : {offre} pour {besoin}"
            raise ValueError(msg)
        point = sp.Matrix(self.particuliere)
        # des parametres en surnombre restent inemployes : la garde suffit
        for parametre, direction in zip(parametres, self.directions, strict=False):
            point += parametre * sp.Matrix(direction)
        return tuple(point)


def par_echelonnement_simple(systeme: SystemeLineaire) -> EnsembleDesSolutions:
    """Les variables des pivots en fonction des variables libres."""
    inconnues = int(systeme.matrice.cols)
    simple = echelonner_simplement(systeme.matrice.row_join(systeme.second_membre))
    pivots = [p for p in colonnes_des_pivots(simple) if p is not None]
    if inconnues in pivots:
        msg = "systeme impossible : une equation se reduit a 0 = 1"
        raise ValueError(msg)
    particuliere = [sp.Integer(0)] * inconnues
    for ligne, pivot in enumerate(pivots):
        particuliere[pivot] = simple[ligne, inconnues]
    directions = []
    for libre in (j for j in range(inconnues) if j not in pivots):
        direction = [sp.Integer(0)] * inconnues
        direction[libre] = sp.Integer(1)
        for ligne, pivot in enumerate(pivots):
            direction[pivot] = -simple[ligne, libre]
        directions.append(tuple(direction))
    return EnsembleDesSolutions(tuple(particuliere), tuple(directions))


def est_l_ensemble_des_solutions(
    systeme: SystemeLineaire,
    parametrisation: Sequence[sp.Expr],
    parametres: Sequence[sp.Symbol],
) -> bool:
    """Vrai si la parametrisation, affine, decrit toutes les solutions et elles seules.

    Elle satisfait le systeme pour toute valeur des parametres, et ses
    parametres independants sont au nombre de `n - rang(A)`.
    """
    point = sp.Matrix(list(parametrisation))
    residu = sp.simplify(systeme.matrice * point - systeme.second_membre)
    if residu != sp.zeros(systeme.matrice.rows, 1):
        return False
    libres = point.jacobian(list(parametres)).rank() if parametres else 0
    return bool(libres == int(systeme.matrice.cols) - systeme.rang())
