"""Extensions, partie II : l'algorithme dual applique aux variables bornees.

« A chaque iteration, soit la variable `x_i`, soit la variable `x_i^e`
est utilisee, mais jamais les deux simultanement. » La marche des
planches 17 a 22 :

1. rendre la base duale admissible, en changeant de variable chaque
   colonne hors base dont le coefficient `a_0j` s'y oppose ;
2. tant qu'une variable de base depasse sa borne, la changer de
   variable : elle devient negative ;
3. tant qu'une variable de base est negative, un pas de l'algorithme
   dual ; puis revenir en 2.

Quand la base est primale admissible et que toutes les bornes sont
verifiees, la solution est optimale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.algorithme import ITERATIONS_MAXIMALES, Issue
from polytech.optimisation.domain.algorithme_dual import pivoter_en_dual
from polytech.optimisation.domain.contrainte_artificielle import indices_a_borner
from polytech.optimisation.domain.criteres_duaux import (
    est_primal_admissible,
    ligne_sans_solution_admissible,
)
from polytech.optimisation.domain.variables_bornees import TableauBorne, complementer

if TYPE_CHECKING:
    from polytech.optimisation.domain.contraintes import Optimisation


@dataclass(frozen=True, slots=True)
class ResolutionBornee:
    """Les etats successifs -- changements de variable et pivots -- et l'arret."""

    etapes: tuple[TableauBorne, ...]
    issue: Issue

    @property
    def valeur(self) -> sp.Expr:
        """`a_00` du dernier tableau."""
        return self.etapes[-1].tableau.valeur

    def solution(self, variables: int) -> sp.Matrix:
        """Les `x_i` d'origine : `u_i - x_i^e` pour les variables changees."""
        final = self.etapes[-1]
        point = final.tableau.solution(variables)
        for indice in final.complementees:
            point[indice - 1, 0] = final.bornes[indice] - point[indice - 1, 0]
        return point

    def _sous(self, indice: int) -> sp.Expr:
        tableau = self.etapes[-1].tableau
        if indice not in tableau.hors_base:
            return sp.Integer(0)
        return -tableau.coefficients[0, tableau.hors_base.index(indice) + 1]

    def variables_duales(self, ecarts: tuple[int, ...]) -> tuple[sp.Expr, ...]:
        """`-a_0j` sous chaque variable d'ecart (en minimisation)."""
        return tuple(self._sous(j) for j in ecarts)

    def duales_des_bornes(self) -> dict[int, sp.Expr]:
        """La variable duale de chaque borne : non nulle si la borne est atteinte."""
        final = self.etapes[-1]
        return {
            j: self._sous(j) if j in final.complementees else sp.Integer(0)
            for j in final.bornes
        }


def _depassee(etat: TableauBorne) -> int | None:
    """Une variable de base au-dela de sa borne superieure, ou `None`."""
    valeurs = zip(etat.tableau.base, etat.tableau.coefficients[1:, 0], strict=True)
    for indice, valeur in valeurs:
        if indice in etat.bornes and valeur > etat.bornes[indice]:
            return indice
    return None


def resoudre_avec_bornes(
    depart: TableauBorne,
    sens: Optimisation,
    iterations_maximales: int = ITERATIONS_MAXIMALES,
) -> ResolutionBornee:
    """L'optimum sous bornes superieures : changements de variable et pivots duaux."""
    etapes = [depart]
    for indice in indices_a_borner(depart.tableau, sens):
        etapes.append(complementer(etapes[-1], indice))
    for _ in range(iterations_maximales):
        courant = etapes[-1]
        depassee = _depassee(courant)
        if depassee is not None:
            etapes.append(complementer(courant, depassee))
        elif est_primal_admissible(courant.tableau):
            return ResolutionBornee(tuple(etapes), Issue.OPTIMALE)
        elif ligne_sans_solution_admissible(courant.tableau) is not None:
            return ResolutionBornee(tuple(etapes), Issue.IMPOSSIBLE)
        else:
            etapes.append(replace(courant, tableau=pivoter_en_dual(courant.tableau)))
    msg = "aucun arret de l'algorithme a variables bornees : cyclage ?"
    raise RuntimeError(msg)
