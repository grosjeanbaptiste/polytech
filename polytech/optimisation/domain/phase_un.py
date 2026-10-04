"""Chapitre II.5 : la phase I, methode des variables artificielles.

On resout le probleme augmente `P^a` par la phase II, puis on conclut
pour le probleme `P` (page 60) :

* `P^a` n'a pas de solution optimale finie : `P` non plus ;
* `P^a` a une solution optimale ou une variable artificielle reste
  positive : `P` est **impossible** ;
* toutes les variables artificielles sont nulles : la solution est une
  solution de base admissible optimale de `P`.

Une variable artificielle sortie de la base n'y rentrera plus : sa
colonne est supprimee (remarque 4 beta). Une variable artificielle
**nulle** restee en base (remarque 2) en est chassee par un pivot non nul
de sa ligne ; s'il n'y en a aucun, la contrainte etait **redondante**,
et sa ligne s'elimine -- c'est ainsi que la methode revele `r(T) < m`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.optimisation.domain.algorithme import (
    ITERATIONS_MAXIMALES,
    Issue,
    Regle,
    pivoter,
)
from polytech.optimisation.domain.artificielles import chasser_les_nulles, sans_colonnes
from polytech.optimisation.domain.criteres import (
    colonne_sans_solution_finie,
    est_optimal,
)
from polytech.optimisation.domain.depart import probleme_augmente
from polytech.optimisation.domain.tableau import TableauSimplexe, tableau_de_la_base

if TYPE_CHECKING:
    from polytech.optimisation.domain.contraintes import Optimisation
    from polytech.optimisation.domain.formes import FormeStandard


@dataclass(frozen=True, slots=True)
class ResolutionComplete:
    """Les tableaux des deux phases, l'arret, et ce que la phase I a revele."""

    tableaux: tuple[TableauSimplexe, ...]
    issue: Issue
    artificielles: tuple[int, ...]
    contraintes_redondantes: int = 0


def _positives(tableau: TableauSimplexe, artificielles: tuple[int, ...]) -> bool:
    return any(
        tableau.coefficients[ligne, 0] > 0
        for ligne, indice in enumerate(tableau.base, start=1)
        if indice in artificielles
    )


def methode_des_variables_artificielles(
    forme: FormeStandard,
    sens: Optimisation,
    regle: Regle = Regle.PLUS_FORT_COEFFICIENT,
) -> ResolutionComplete:
    """Resout `P` par `P^a` : optimal, non borne, ou impossible."""
    augmente, base = probleme_augmente(forme, sens)
    artificielles = tuple(range(forme.matrice.cols + 1, augmente.matrice.cols + 1))
    tableaux = [tableau_de_la_base(augmente, base)]
    redondantes = 0
    for _ in range(ITERATIONS_MAXIMALES):
        courant = tableaux[-1]
        if colonne_sans_solution_finie(courant, sens) is not None:
            return ResolutionComplete(tuple(tableaux), Issue.NON_BORNEE, artificielles)
        if not est_optimal(courant, sens):
            tableaux.append(sans_colonnes(pivoter(courant, sens, regle), artificielles))
            continue
        if _positives(courant, artificielles):
            return ResolutionComplete(tuple(tableaux), Issue.IMPOSSIBLE, artificielles)
        nettoye, eliminees = chasser_les_nulles(courant, artificielles)
        if nettoye == courant:
            return ResolutionComplete(
                tuple(tableaux), Issue.OPTIMALE, artificielles, redondantes
            )
        redondantes += eliminees
        tableaux.append(nettoye)
    msg = f"aucun arret apres {ITERATIONS_MAXIMALES} iterations en phase I : cyclage ?"
    raise RuntimeError(msg)
