"""Chapitre I.3.4, suite : le polyedre est-il borne, et sinon ou part-il.

Un polyedre convexe non borne contient un **rayon** : il existe un point
`x` de `P` et une direction `d` tels que `x + lambda d` reste dans `P`
pour tout `lambda >= 0`. La direction verifie `alpha_i d <= 0` pour
chaque demi-espace. Les **rayons extremes** sont ceux qui longent une
arete : `n - 1` hyperplans independants y sont paralleles a `d`.

Le raisonnement suppose que le polyedre a un sommet. Un polyedre dont
les normales ne sont pas de rang `n` contient une droite entiere -- ou
bien il est vide -- et n'a pas de sommet : la question est alors
refusee, plutot que de repondre a cote.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.optimisation.domain.polyedres import Polyedre


def _exige_un_sommet_possible(polyedre: Polyedre) -> None:
    normales = sp.Matrix([list(d.normale) for d in polyedre.demi_espaces])
    if normales.rank() < polyedre.dimension:
        msg = (
            "polyedre sans sommet : il contient une droite ou il est vide "
            f"(rang {normales.rank()} en dimension {polyedre.dimension})"
        )
        raise ValueError(msg)


def _est_une_direction(polyedre: Polyedre, direction: sp.Matrix) -> bool:
    return all(
        sp.Matrix([list(d.normale)]).dot(direction) <= 0 for d in polyedre.demi_espaces
    )


def rayons_extremes(polyedre: Polyedre) -> tuple[sp.Matrix, ...]:
    """Les directions extremes de `P`, normalisees par `somme |d_i| = 1`."""
    _exige_un_sommet_possible(polyedre)
    rayons: list[sp.Matrix] = []
    for choix in combinations(polyedre.demi_espaces, polyedre.dimension - 1):
        noyau = sp.Matrix([list(d.normale) for d in choix]).nullspace()
        if len(noyau) != 1:
            continue
        for sens in (1, -1):
            direction = sens * noyau[0] / sum(abs(v) for v in noyau[0])
            if _est_une_direction(polyedre, direction) and direction not in rayons:
                rayons.append(direction)
    return tuple(rayons)


def est_borne(polyedre: Polyedre) -> bool:
    """Vrai si le polyedre ne contient aucun rayon."""
    return not rayons_extremes(polyedre)


def est_un_polytope(polyedre: Polyedre) -> bool:
    """Un polyedre convexe borne et non vide."""
    return est_borne(polyedre) and bool(polyedre.sommets())
