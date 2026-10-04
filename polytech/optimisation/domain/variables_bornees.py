"""Extensions, partie II : les variables bornees `l_i <= x_i <= u_i`.

Une borne **inferieure** s'efface par translation, `x' = x - l` : la
fonction economique gagne la constante `c l`, les seconds membres perdent
`T l`. Seules les bornes superieures demandent donc un traitement.

Les ecrire en contraintes, `x_i + x_i^e = u_i`, doublerait le nombre de
variables. Les deux variables `x_i` et `x_i^e` sont **symetriques** --
connaitre l'une, c'est connaitre l'autre -- et le tableau n'en porte
jamais qu'une. Passer de l'une a l'autre, `x_i = u_i - x_i^e`, est un
**changement de variable** qui se lit sur le tableau :

* hors base, la colonne change de signe et la colonne des seconds membres
  perd `u_i` fois l'ancienne colonne ;
* en base, la ligne change de signe et son second membre devient
  `u_i - a_i0`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte
from polytech.optimisation.domain.tableau import TableauSimplexe

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polytech.optimisation.domain.formulation import ProgrammeGeneral


@dataclass(frozen=True, slots=True)
class TableauBorne:
    """Un tableau, les bornes superieures, et les variables lues par leur `x^e`."""

    tableau: TableauSimplexe
    bornes: Mapping[int, sp.Expr]
    complementees: frozenset[int] = field(default_factory=frozenset)


def complementer(etat: TableauBorne, indice: int) -> TableauBorne:
    """Le changement de variable `x_i = u_i - x_i^e` (ou son inverse)."""
    if indice not in etat.bornes:
        msg = f"x_{indice} est sans borne superieure : pas de changement de variable"
        raise ValueError(msg)
    borne = sp.sympify(etat.bornes[indice])
    tableau = etat.tableau
    coefficients = tableau.coefficients.copy()
    if indice in tableau.base:
        ligne = tableau.base.index(indice) + 1
        coefficients[ligne, :] = -coefficients[ligne, :]
        coefficients[ligne, 0] = borne - tableau.coefficients[ligne, 0]
    else:
        colonne = tableau.hors_base.index(indice) + 1
        coefficients[:, 0] = coefficients[:, 0] - borne * coefficients[:, colonne]
        coefficients[:, colonne] = -coefficients[:, colonne]
    return replace(
        etat,
        tableau=TableauSimplexe(tableau.base, tableau.hors_base, coefficients),
        complementees=etat.complementees ^ {indice},
    )


def translater_les_bornes_inferieures(
    programme: ProgrammeGeneral,
    bornes: Sequence[object],
) -> tuple[ProgrammeGeneral, sp.Expr]:
    """Le programme en `x' = x - l`, et la constante `c l` a rendre a `z`."""
    contraintes = tuple(
        Contrainte(c.coefficients, c.sens, c.second_membre - c.premier_membre(bornes))
        for c in programme.contraintes
    )
    return replace(programme, contraintes=contraintes), programme.valeur(bornes)
