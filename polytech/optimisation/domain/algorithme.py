"""Chapitre II.4.2 : l'algorithme simplexe, phase II.

A partir du tableau d'une solution de base admissible :

1. si aucune variable hors base n'ameliore `z`, la solution est
   optimale -- STOP ;
2. si une variable ameliore `z` sans qu'aucun pivot positif ne la borne,
   il n'y a pas de solution optimale finie -- STOP ;
3. sinon, changer de base : la variable entrante est celle du plus fort
   coefficient, la sortante celle du plus petit rapport, et l'on
   recommence.

Sans degenerescence, `z` s'ameliore strictement a chaque iteration, aucun
sommet n'est visite deux fois, et l'algorithme converge (propriete
II.4.1). Avec, il faut une regle : lexicographique par defaut, ou celle
de Bland (II.6.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

from polytech.optimisation.domain.criteres import (
    colonne_sans_solution_finie,
    est_optimal,
    variable_entrante,
)
from polytech.optimisation.domain.pivot import changer_de_base
from polytech.optimisation.domain.regles import (
    entrante_de_bland,
    sortante_de_bland,
    sortante_lexicographique,
)

if TYPE_CHECKING:
    from polytech.optimisation.domain.contraintes import Optimisation
    from polytech.optimisation.domain.tableau import TableauSimplexe

ITERATIONS_MAXIMALES: Final = 200
"""Garde-fou : les deux regles convergent, une boucle sans fin serait un defaut."""


class Issue(Enum):
    """Les arrets de l'algorithme ; le troisieme n'apparait qu'en phase I."""

    OPTIMALE = "solution optimale"
    NON_BORNEE = "pas de solution optimale finie"
    IMPOSSIBLE = "probleme impossible : aucune solution admissible"


class Regle(Enum):
    """Comment choisir la variable entrante et departager les sortantes."""

    PLUS_FORT_COEFFICIENT = "plus fort coefficient, sortante lexicographique"
    BLAND = "plus petits indices"


@dataclass(frozen=True, slots=True)
class Resolution:
    """Les tableaux successifs, l'arret, et la colonne qui le cause s'il y a lieu."""

    tableaux: tuple[TableauSimplexe, ...]
    issue: Issue
    colonne_non_bornee: int | None = None


def pivoter(
    tableau: TableauSimplexe, sens: Optimisation, regle: Regle
) -> TableauSimplexe:
    """Une iteration : le tableau de la base suivante, selon la regle choisie."""
    if regle is Regle.BLAND:
        entrante = entrante_de_bland(tableau, sens)
        return changer_de_base(tableau, sortante_de_bland(tableau, entrante), entrante)
    entrante = variable_entrante(tableau, sens)
    variables = max((*tableau.base, *tableau.hors_base))
    sortante = sortante_lexicographique(tableau, entrante, variables)
    return changer_de_base(tableau, sortante, entrante)


def phase_deux(
    depart: TableauSimplexe,
    sens: Optimisation,
    regle: Regle = Regle.PLUS_FORT_COEFFICIENT,
    iterations_maximales: int = ITERATIONS_MAXIMALES,
) -> Resolution:
    """Deroule l'algorithme depuis une solution de base admissible."""
    tableaux = [depart]
    for _ in range(iterations_maximales):
        courant = tableaux[-1]
        if est_optimal(courant, sens):
            return Resolution(tuple(tableaux), Issue.OPTIMALE)
        colonne = colonne_sans_solution_finie(courant, sens)
        if colonne is not None:
            return Resolution(tuple(tableaux), Issue.NON_BORNEE, colonne)
        tableaux.append(pivoter(courant, sens, regle))
    msg = f"aucun arret apres {iterations_maximales} iterations : cyclage ?"
    raise RuntimeError(msg)
