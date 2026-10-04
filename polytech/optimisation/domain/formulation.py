"""Chapitre I.1 : la forme generale d'un programme lineaire.

La forme **generale** optimise `z = c x` -- au minimum ou au maximum --
sous des contraintes reelles de trois sens (`<=`, `>=`, `=`), avec des
variables non negatives ou **de signe quelconque**. Le cours la ramene a
deux formes plus compactes :

* la forme **canonique** : `min z = c x`, `T x >= d`, `x >= 0` ;
* la forme **standard** : `min z = c x`, `T x = d`, `x >= 0`.

Ces deux formes sont dans `formes` ; ce module porte la forme generale
et les gestes sur ses variables et son objectif.

Trois gestes y menent, et chacun est une methode : maximiser `z`
equivaut a minimiser `-z` ; une inegalite `<=` se retourne en la
multipliant par `-1`, une egalite se dedouble en deux `>=` ; une variable
libre se decompose en `x_j = x_j+ - x_j-`, ou bien se translate en
`x_j = x_j' - xi` -- une seule variable de plus pour toutes les libres.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.contraintes import (
    Contrainte,
    Optimisation,
    exacts,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ProgrammeGeneral:
    """Un programme lineaire sous la forme la plus generale du cours."""

    optimisation: Optimisation
    objectif: tuple[sp.Expr, ...]
    contraintes: tuple[Contrainte, ...]
    libres: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Refuse une contrainte de mauvaise longueur et une libre inexistante."""
        object.__setattr__(self, "objectif", exacts(self.objectif))
        n = len(self.objectif)
        for contrainte in self.contraintes:
            if len(contrainte.coefficients) != n:
                msg = f"{len(contrainte.coefficients)} coefficients pour {n} variables"
                raise ValueError(msg)
        hors = sorted(j for j in self.libres if not 0 <= j < n)
        if hors:
            msg = f"variable libre hors du programme : {hors} pour {n} variables"
            raise ValueError(msg)

    def valeur(self, point: Sequence[object]) -> sp.Expr:
        """`z = c x` au point donne."""
        return sp.Add(
            *(c * sp.sympify(x) for c, x in zip(self.objectif, point, strict=True))
        )

    def a_variables_non_negatives(self) -> ProgrammeGeneral:
        """`x_j = x_j+ - x_j-` : x_j+ garde sa colonne, x_j- s'ajoute a la fin."""
        programme = replace(self, libres=frozenset())
        for j in sorted(self.libres):
            colonne = [-c.coefficients[j] for c in self.contraintes]
            programme = _avec_colonne(programme, -self.objectif[j], colonne)
        return programme

    def translater_les_libres(self) -> ProgrammeGeneral:
        """`x_j = x_j' - xi` pour toutes les libres : une seule variable de plus."""
        libres = sorted(self.libres)
        colonne = [
            -sp.Add(*(c.coefficients[j] for j in libres)) for c in self.contraintes
        ]
        cout = -sp.Add(*(self.objectif[j] for j in libres))
        return _avec_colonne(replace(self, libres=frozenset()), cout, colonne)

    def en_minimisation(self) -> ProgrammeGeneral:
        """`max z` equivaut a `min -z`."""
        if self.optimisation is Optimisation.MIN:
            return self
        return replace(
            self,
            optimisation=Optimisation.MIN,
            objectif=tuple(-c for c in self.objectif),
        )


def _avec_colonne(
    programme: ProgrammeGeneral,
    cout: sp.Expr,
    colonne: Sequence[sp.Expr],
) -> ProgrammeGeneral:
    """Le programme augmente d'une variable : son cout et sa colonne."""
    contraintes = tuple(
        replace(c, coefficients=(*c.coefficients, a))
        for c, a in zip(programme.contraintes, colonne, strict=True)
    )
    return replace(
        programme,
        objectif=(*programme.objectif, cout),
        contraintes=contraintes,
    )
