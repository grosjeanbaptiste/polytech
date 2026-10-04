"""Chapitre II.5 : la phase I, methode des variables artificielles.

On resout le probleme augmente `P^a` par la phase II, puis on conclut
pour le probleme `P` (page 60) :

* une variable artificielle reste positive sans qu'aucun terme en `M` ne
  puisse plus la reduire : `P` est **impossible** ;
* toutes les variables artificielles sont nulles : la suite est une
  phase II ordinaire sur `P`, qui s'arrete a l'optimum ou constate que
  `P` n'a pas de solution optimale finie.

**Un ecart au cours, etabli.** Le cours ecrit page 60 que si `P^a` n'a
pas de solution optimale finie, « les variables artificielles sont
necessairement toutes nulles » et `P` est non borne. C'est faux en
general : `max 2 x1 + 2 x2`, `-x1 + x2 >= 1`, `x1 - x2 >= 1` -- le
troisieme couple de sa propre page 95 -- est impossible, et son `P^a`
n'est pas borne le long de `x1 = x2`, artificielles a 1. La conclusion
se lit donc **d'abord** sur les termes en `M`, comme la remarque 4 gamma
de la page 62 le dit d'ailleurs : tant que la phase I n'est pas
terminee, ce sont eux qui decident.

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
    colonnes_reduisant_la_penalite,
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
        if _positives(courant, artificielles):
            if not colonnes_reduisant_la_penalite(courant, sens):
                return ResolutionComplete(
                    tuple(tableaux), Issue.IMPOSSIBLE, artificielles
                )
            tableaux.append(sans_colonnes(pivoter(courant, sens, regle), artificielles))
            continue
        nettoye, eliminees = chasser_les_nulles(courant, artificielles)
        if nettoye != courant:
            redondantes += eliminees
            tableaux.append(nettoye)
            continue
        if est_optimal(courant, sens):
            return ResolutionComplete(
                tuple(tableaux), Issue.OPTIMALE, artificielles, redondantes
            )
        if colonne_sans_solution_finie(courant, sens) is not None:
            return ResolutionComplete(
                tuple(tableaux), Issue.NON_BORNEE, artificielles, redondantes
            )
        tableaux.append(pivoter(courant, sens, regle))
    msg = f"aucun arret apres {ITERATIONS_MAXIMALES} iterations en phase I : cyclage ?"
    raise RuntimeError(msg)
