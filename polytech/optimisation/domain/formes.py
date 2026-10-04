"""Chapitre I.1 : les deux formes equivalentes d'un programme lineaire.

* la forme **canonique** : `min z = c x`, `T x >= d`, `x >= 0` -- une
  inegalite `<=` se retourne en la multipliant par `-1`, une egalite se
  dedouble en deux `>=` ;
* la forme **standard** : `min z = c x`, `T x = d`, `x >= 0` -- chaque
  inegalite recoit une **variable d'ecart** non negative, ajoutee a la fin
  dans l'ordre des contraintes ; une egalite n'en recoit pas.

Les deux partent du programme ramene a des variables non negatives et a
la minimisation, comme le cours le fait page 25.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Sens

if TYPE_CHECKING:
    from polytech.optimisation.domain.formulation import ProgrammeGeneral


@dataclass(frozen=True, slots=True)
class FormeStandard:
    """`min z = c x`, `T x = d`, `x >= 0` : couts, matrice technologique, `d`."""

    couts: sp.Matrix
    matrice: sp.Matrix
    seconds_membres: sp.Matrix


def forme_canonique(programme: ProgrammeGeneral) -> ProgrammeGeneral:
    """`min z = c x`, `T x >= d`, `x >= 0`."""
    base = programme.a_variables_non_negatives().en_minimisation()
    contraintes: list[Contrainte] = []
    for c in base.contraintes:
        if c.sens is Sens.INFERIEUR:
            contraintes.append(c.retournee())
        elif c.sens is Sens.EGAL:
            contraintes += [
                replace(c, sens=Sens.SUPERIEUR),
                replace(c.retournee(), sens=Sens.SUPERIEUR),
            ]
        else:
            contraintes.append(c)
    return replace(base, contraintes=tuple(contraintes))


def forme_standard(programme: ProgrammeGeneral) -> FormeStandard:
    """`min z = c x`, `T x = d`, `x >= 0` : un ecart par inegalite, dans l'ordre."""
    base = programme.a_variables_non_negatives().en_minimisation()
    signes = {Sens.INFERIEUR: 1, Sens.SUPERIEUR: -1}
    inegalites = [i for i, c in enumerate(base.contraintes) if c.sens in signes]
    ecarts = sp.zeros(len(base.contraintes), len(inegalites))
    for colonne, i in enumerate(inegalites):
        ecarts[i, colonne] = signes[base.contraintes[i].sens]
    matrice = sp.Matrix([list(c.coefficients) for c in base.contraintes]).row_join(
        ecarts
    )
    return FormeStandard(
        couts=sp.Matrix([[*base.objectif, *([0] * len(inegalites))]]),
        matrice=matrice,
        seconds_membres=sp.Matrix([c.second_membre for c in base.contraintes]),
    )
