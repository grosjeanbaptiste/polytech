"""Chapitre XIV.2.2 B : les relaxations qui donnent une fonction d'evaluation.

Relaxer, c'est elargir l'ensemble des solutions pour obtenir un probleme
plus simple, dont l'optimum **minore** celui du probleme de depart. Trois
classes, pour un polyedre `{x | T1 x <= d1, T2 x <= d2}` :

* la relaxation **generale** : supprimer les contraintes `T2 x <= d2` ;
* la relaxation **lineaire** : supprimer l'integrite (voir `entiers`) ;
* la relaxation **lagrangienne** : supprimer les contraintes `T2 x <= d2`
  mais les faire payer dans la fonction economique, par des
  multiplicateurs `lambda >= 0` :

      v(lambda) = min F(x) + lambda (T2 x - d2).

  Sur une solution admissible `T2 x - d2 <= 0`, donc `v(lambda) <= F`
  quel que soit `lambda` : chaque `lambda` donne une borne, et chercher la
  meilleure est une forme de probleme dual.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.contraintes import Optimisation, Sens

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polytech.optimisation.domain.formulation import ProgrammeGeneral


def relaxation_generale(
    programme: ProgrammeGeneral,
    relachees: Sequence[int],
) -> ProgrammeGeneral:
    """Le programme sans les contraintes d'indices (de 0) `relachees`."""
    gardees = tuple(
        c for i, c in enumerate(programme.contraintes) if i not in relachees
    )
    return replace(programme, contraintes=gardees)


def relaxation_lagrangienne(
    programme: ProgrammeGeneral,
    multiplicateurs: Mapping[int, object],
) -> tuple[ProgrammeGeneral, sp.Expr]:
    """Le programme relaxe et la constante `-lambda d2` a rendre a sa valeur."""
    if programme.optimisation is not Optimisation.MIN:
        msg = "la relaxation lagrangienne s'ecrit pour une minimisation"
        raise ValueError(msg)
    objectif = list(programme.objectif)
    constante = sp.Integer(0)
    for indice, brut in multiplicateurs.items():
        poids = sp.sympify(brut)
        contrainte = programme.contraintes[indice]
        if poids < 0:
            msg = f"multiplicateur negatif pour la contrainte {indice} : {poids}"
            raise ValueError(msg)
        if contrainte.sens is Sens.EGAL:
            msg = f"la contrainte {indice} est une egalite : multiplicateur libre"
            raise ValueError(msg)
        signe = 1 if contrainte.sens is Sens.INFERIEUR else -1
        for j, a in enumerate(contrainte.coefficients):
            objectif[j] += signe * poids * a
        constante -= signe * poids * contrainte.second_membre
    relaxe = relaxation_generale(programme, tuple(multiplicateurs))
    return replace(relaxe, objectif=tuple(objectif)), constante
