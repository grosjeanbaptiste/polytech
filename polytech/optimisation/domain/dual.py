"""Chapitre IV.1 et IV.2 : le probleme dual, et ce qui le lie au primal.

A tout programme lineaire, le **primal**, la dualite en associe un autre,
le **dual**. Par convention, les inegalites d'une minimisation s'ecrivent
`>=`, celles d'une maximisation `<=`. Alors (page 91) :

* a une minimisation correspond une maximisation, et inversement ;
* a chaque contrainte primale correspond une variable duale, non
  negative si la contrainte est une inegalite, de signe quelconque si
  c'est une egalite ;
* a chaque variable primale correspond une contrainte duale, inegalite
  si la variable est non negative, egalite si elle est libre ;
* les coefficients de la fonction economique et les seconds membres
  s'echangent, et la matrice se transpose.

L'operation est **involutive**. Deux proprietes s'ensuivent : la dualite
**faible** -- chaque solution admissible de l'un borne l'autre -- et la
dualite **forte** -- s'il y a un optimum fini, il y en a deux, et ils
sont egaux. D'ou le theoreme fondamental : sur neuf situations
concevables, quatre seulement existent.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre

if TYPE_CHECKING:
    from collections.abc import Sequence


class Situation(Enum):
    """Les situations qu'un couple primal-dual peut presenter (corollaire 5)."""

    DEUX_OPTIMUMS_FINIS = "les deux problemes ont des solutions optimales finies"
    NON_BORNE_ET_IMPOSSIBLE = "le primal est non borne, le dual impossible"
    IMPOSSIBLE_ET_NON_BORNE = "le primal est impossible, le dual non borne"
    DEUX_IMPOSSIBLES = "les deux problemes sont impossibles"


def _sens_conventionnel(optimisation: Optimisation) -> Sens:
    return Sens.SUPERIEUR if optimisation is Optimisation.MIN else Sens.INFERIEUR


def sous_la_convention(programme: ProgrammeGeneral) -> ProgrammeGeneral:
    """Le meme programme, ses inegalites ecrites `>=` (min) ou `<=` (max)."""
    voulu = _sens_conventionnel(programme.optimisation)
    contraintes = tuple(
        c if c.sens in (voulu, Sens.EGAL) else c.retournee()
        for c in programme.contraintes
    )
    return replace(programme, contraintes=contraintes)


def dual_de(programme: ProgrammeGeneral) -> ProgrammeGeneral:
    """Le probleme dual, par les regles de la page 91."""
    primal = sous_la_convention(programme)
    inverse = (
        Optimisation.MAX
        if primal.optimisation is Optimisation.MIN
        else Optimisation.MIN
    )
    sens = _sens_conventionnel(inverse)
    contraintes = tuple(
        Contrainte(
            tuple(c.coefficients[j] for c in primal.contraintes),
            Sens.EGAL if j in primal.libres else sens,
            cout,
        )
        for j, cout in enumerate(primal.objectif)
    )
    return ProgrammeGeneral(
        optimisation=inverse,
        objectif=tuple(c.second_membre for c in primal.contraintes),
        contraintes=contraintes,
        libres=frozenset(
            i for i, c in enumerate(primal.contraintes) if c.sens is Sens.EGAL
        ),
    )


def ecart_de_dualite(
    primal: ProgrammeGeneral,
    x: Sequence[object],
    u: Sequence[object],
) -> sp.Expr:
    """L'ecart entre les deux fonctions economiques ; jamais negatif (propriete 1)."""
    difference = primal.valeur(x) - dual_de(primal).valeur(u)
    return sp.simplify(
        difference if primal.optimisation is Optimisation.MIN else -difference
    )


def situation_du_couple(primal: ProgrammeGeneral) -> Situation:
    """Laquelle des quatre situations du theoreme fondamental le couple presente."""
    issues = (resoudre(primal).issue, resoudre(dual_de(primal)).issue)
    situations = {
        (Issue.OPTIMALE, Issue.OPTIMALE): Situation.DEUX_OPTIMUMS_FINIS,
        (Issue.NON_BORNEE, Issue.IMPOSSIBLE): Situation.NON_BORNE_ET_IMPOSSIBLE,
        (Issue.IMPOSSIBLE, Issue.NON_BORNEE): Situation.IMPOSSIBLE_ET_NON_BORNE,
        (Issue.IMPOSSIBLE, Issue.IMPOSSIBLE): Situation.DEUX_IMPOSSIBLES,
    }
    if issues not in situations:
        msg = f"couple contraire au theoreme fondamental de la dualite : {issues}"
        raise RuntimeError(msg)
    return situations[issues]
