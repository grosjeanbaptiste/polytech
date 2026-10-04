"""Chapitre XIV.1.1 : les problemes lineaires en variables discretes.

Trois problemes sur le meme polyedre `D = {x | T x >= d, x >= 0}` :

* (LP) : `x` dans `D`, variables continues ;
* (ILP) : toutes les variables entieres, `x` dans `S = D n Z^n` ;
* (MILP) : une partie seulement des variables entieres -- le probleme
  mixte.

L'optimum de (ILP) ou de (MILP) n'est en general **pas** un sommet de
`D` : on perd la caracterisation par les solutions de base, et c'est ce
qui rend ces problemes d'une autre nature que (LP). La **relaxation
lineaire** oublie l'integrite ; elle borne le probleme entier sans le
resoudre.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.formulation import ProgrammeGeneral


class Nature(Enum):
    """Ce que les contraintes d'integrite font du probleme."""

    LP = "lineaire, en variables continues"
    ILP = "lineaire, totalement en variables entieres"
    MILP = "lineaire mixte"


@dataclass(frozen=True, slots=True)
class ProgrammeEntier:
    """Un programme lineaire et les indices (de 0) de ses variables entieres."""

    programme: ProgrammeGeneral
    entieres: frozenset[int]

    def __post_init__(self) -> None:
        """Refuse une contrainte d'integrite sur une variable qui n'existe pas."""
        n = len(self.programme.objectif)
        hors = sorted(j for j in self.entieres if not 0 <= j < n)
        if hors:
            msg = f"variable entiere hors du programme : {hors} pour {n} variables"
            raise ValueError(msg)

    @property
    def nature(self) -> Nature:
        """(LP), (ILP) ou (MILP), selon le nombre de variables astreintes."""
        if not self.entieres:
            return Nature.LP
        complet = len(self.entieres) == len(self.programme.objectif)
        return Nature.ILP if complet else Nature.MILP

    def relaxation_lineaire(self) -> ProgrammeGeneral:
        """Le meme programme, les contraintes d'integrite supprimees."""
        return self.programme

    def est_entier(self, point: Sequence[object]) -> bool:
        """Vrai si chaque variable astreinte prend une valeur entiere."""
        return all(sp.sympify(point[j]).is_integer for j in self.entieres)

    def est_admissible(self, point: Sequence[object]) -> bool:
        """Contraintes reelles, non-negativite, integrite."""
        libres = self.programme.libres
        positif = all(
            sp.sympify(x) >= 0 for j, x in enumerate(point) if j not in libres
        )
        reelles = all(c.est_satisfaite(point) for c in self.programme.contraintes)
        return positif and reelles and self.est_entier(point)


def points_entiers(
    programme: ProgrammeGeneral,
    boite: Sequence[tuple[int, int]],
) -> frozenset[tuple[int, ...]]:
    """`S = D n Z^n`, enumere dans une boite : le chemin naif, pour verifier."""
    entier = ProgrammeEntier(programme, frozenset(range(len(programme.objectif))))
    axes = (range(bas, haut + 1) for bas, haut in boite)
    return frozenset(point for point in product(*axes) if entier.est_admissible(point))
