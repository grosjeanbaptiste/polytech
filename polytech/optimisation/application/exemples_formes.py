"""Exemples resolus, planches 3 a 7 : les formes d'un programme lineaire.

Un probleme a trois variables, dont deux de signe quelconque, est ecrit
sous forme generale, puis canonique, puis standard. Deux facons de rendre
positives les variables libres : les dedoubler, ou les translater toutes
d'une meme quantite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import forme_canonique
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.algorithme import Issue

GENERAL: Final = ProgrammeGeneral(
    Optimisation.MAX,
    (1, -2, 4),
    (
        Contrainte((1, 3, 2), Sens.INFERIEUR, 10),
        Contrainte((2, -1, 6), Sens.SUPERIEUR, 5),
        Contrainte((1, 2, 3), Sens.EGAL, 6),
    ),
    libres=frozenset({0, 2}),
)
Ligne = tuple[tuple[sp.Expr, ...], str, sp.Expr]


def variables(*, decomposees: bool) -> int:
    """Le nombre de variables, selon la facon de traiter les libres."""
    if decomposees:
        return len(GENERAL.a_variables_non_negatives().objectif)
    return len(GENERAL.translater_les_libres().objectif)


def translatee() -> tuple[tuple[sp.Expr, ...], tuple[Ligne, ...]]:
    """La minimisation en `x1'`, `x2`, `x3'` et `xbar`, contraintes inchangees."""
    programme = GENERAL.translater_les_libres().en_minimisation()
    lignes = tuple(
        (c.coefficients, c.sens.value, c.second_membre) for c in programme.contraintes
    )
    return programme.objectif, lignes


def canonique() -> tuple[tuple[tuple[sp.Expr, ...], sp.Expr], ...]:
    """Les contraintes `>=` de la forme canonique, apres translation."""
    programme = forme_canonique(GENERAL.translater_les_libres())
    return tuple((c.coefficients, c.second_membre) for c in programme.contraintes)


def issue() -> Issue:
    """Ce que le simplexe conclut sur ce probleme."""
    return resoudre(GENERAL).issue


def point_du_rayon(t: object) -> tuple[sp.Expr, ...]:
    """`(6 - 3t, 0, t)` : la contrainte d'egalite tenue, `x3` croissant."""
    parametre = sp.sympify(t)
    return 6 - 3 * parametre, sp.Integer(0), parametre


def est_admissible(point: Sequence[sp.Expr]) -> bool:
    """Vrai si le point respecte les trois contraintes et `x2 >= 0`."""
    ecarts = [c.ecart(point) for c in GENERAL.contraintes]
    egalite = ecarts[2] == 0
    return bool(egalite and ecarts[0] >= 0 and ecarts[1] >= 0 and point[1] >= 0)


def valeur(point: Sequence[sp.Expr]) -> sp.Expr:
    """L'objectif `x1 - 2x2 + 4x3` en ce point."""
    return GENERAL.valeur(point)
