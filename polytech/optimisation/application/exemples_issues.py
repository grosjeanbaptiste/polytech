"""Exemples resolus, planches 31 a 44 : les trois issues du simplexe.

Un programme lineaire a un optimum fini, ou n'est pas borne, ou n'a aucun
point admissible. Les planches montrent comment chaque cas se lit dans un
tableau.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.application.exemples_sommets import PREMIER
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import FormeStandard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.tableau import tableau_de_la_base

if TYPE_CHECKING:
    from polytech.optimisation.domain.algorithme import Issue

INF, SUP = Sens.INFERIEUR, Sens.SUPERIEUR
M: Final = sp.Symbol("M", positive=True)
_LIGNES: Final = ((1, 1), 8), ((-2, 3), 6), ((1, -1), 2)


def _retourne(*sens: Sens) -> ProgrammeGeneral:
    """L'exemple I, chaque contrainte prenant le sens donne."""
    contraintes = tuple(
        Contrainte(a, s, d) for (a, d), s in zip(_LIGNES, sens, strict=True)
    )
    return ProgrammeGeneral(Optimisation.MAX, (6, 5), contraintes)


PROGRAMMES: Final = {
    "I": PREMIER,
    "II": ProgrammeGeneral(
        Optimisation.MIN,
        (-3, 5),
        (Contrainte((-2, 3), INF, 6), Contrainte((1, -4), INF, 4)),
    ),
    "I retourne": _retourne(SUP, INF, INF),
    "II retourne": _retourne(INF, SUP, SUP),
    "rayon": ProgrammeGeneral(
        Optimisation.MIN,
        (-2, 8),
        (Contrainte((-2, 3), INF, 6), Contrainte((1, -4), INF, 4)),
    ),
}
"""Planches 24, 31, 33, 37 et 44."""


def issue(nom: str) -> Issue:
    """Ce que le simplexe conclut sur le programme de ce nom."""
    return resoudre(PROGRAMMES[nom]).issue


def optimum(nom: str) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """La valeur et le point optimaux du programme de ce nom."""
    return optimum_fini(PROGRAMMES[nom])


def optimum_du_probleme_penalise() -> tuple[sp.Expr, sp.Expr]:
    """Planche 40 : la valeur de `Pa` a son optimum, et l'artificielle qui reste.

    Colonnes : `x1 x2 x1e x2e x3e x1a x2a` ; la base finale est
    `(x1, x2, x2a)`.
    """
    matrice = sp.Matrix(
        [[1, 1, 1, 0, 0, 0, 0], [-2, 3, 0, -1, 0, 1, 0], [1, -1, 0, 0, -1, 0, 1]]
    )
    couts = sp.Matrix([[-6, -5, 0, 0, 0, M, M]])
    penalise = FormeStandard(couts, matrice, sp.Matrix([8, 6, 2]))
    final = tableau_de_la_base(penalise, (1, 2, 7))
    return final.valeur, final.coefficients[3, 0]


def sur_le_rayon(t: object) -> sp.Expr:
    """La valeur de l'objectif en `(4 + 4t, t)`, le long du rayon de la planche 44."""
    parametre = sp.sympify(t)
    return PROGRAMMES["rayon"].valeur((4 + 4 * parametre, parametre))
