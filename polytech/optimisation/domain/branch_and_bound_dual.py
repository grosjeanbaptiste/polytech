"""Chapitre XIV.3 : l'algorithme de separation et d'evaluation des logiciels.

Chaque noeud est un tableau simplexe. Son **evaluation** est la valeur de
sa relaxation lineaire, obtenue en reoptimisant par l'algorithme dual le
tableau herite du parent -- quelques pivots, et non une resolution
entiere. Le **cheminement** est en profondeur d'abord : la regle de
selection choisit la variable et la branche examinee la premiere,
l'autre branche l'est au retour en arriere.

Quand tous les coefficients de la fonction economique sont entiers, `z`
est entiere en tout point admissible : l'evaluation d'un noeud peut etre
arrondie par exces, ce qui sonde davantage (page 310).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.algorithme_dual import phase_deux_duale
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.penalites import (
    Regle,
    choisir_la_separation,
    penalites,
)
from polytech.optimisation.domain.separation_evaluation import (
    Cheminement,
    Procedure,
    explorer,
)
from polytech.optimisation.domain.separation_sur_tableau import (
    Branche,
    NoeudTableau,
    partie_decimale,
    separer_le_tableau,
)
from polytech.optimisation.domain.sondage import Evaluation, Exploration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.tableau import TableauSimplexe

type Point = tuple[sp.Expr, ...]


@dataclass(frozen=True, slots=True)
class OptimumDuNoeud:
    """Le tableau reoptimise d'un noeud, et son point dans les variables d'origine."""

    noeud: NoeudTableau
    point: Point


def _reoptimiser(noeud: NoeudTableau) -> NoeudTableau | None:
    resolution = phase_deux_duale(noeud.tableau, Optimisation.MIN)
    if resolution.issue is Issue.IMPOSSIBLE:
        return None
    return NoeudTableau(resolution.tableaux[-1], noeud.substitutions, noeud.description)


def _point(noeud: NoeudTableau) -> Point:
    variables = max((*noeud.tableau.base, *noeud.tableau.hors_base))
    brut = noeud.tableau.solution(variables)
    return tuple(
        noeud.valeur_d_origine(j, brut[j - 1, 0]) for j in range(1, variables + 1)
    )


@dataclass(frozen=True, slots=True)
class Reoptimisation:
    """L'evaluation et la separation d'un noeud, pour une regle de selection."""

    entieres: frozenset[int]
    regle: Regle
    couts_entiers: bool = False

    def arrondir(self, valeur: sp.Expr) -> sp.Expr:
        """Par exces si `z` est entiere en tout point admissible (page 310)."""
        return sp.ceiling(valeur) if self.couts_entiers else valeur

    def evaluer(self, noeud: NoeudTableau) -> Evaluation[OptimumDuNoeud]:
        """`v_i` : la valeur de la relaxation lineaire, par l'algorithme dual."""
        optimal = _reoptimiser(noeud)
        if optimal is None:
            return Evaluation(None, None, admissible=False)
        point = _point(optimal)
        entier = all(partie_decimale(point[j - 1]) == 0 for j in self.entieres)
        valeur = self.arrondir(optimal.tableau.valeur)
        return Evaluation(valeur, OptimumDuNoeud(optimal, point), admissible=entier)

    def separer(
        self,
        _noeud: NoeudTableau,
        evaluation: Evaluation[OptimumDuNoeud],
    ) -> Sequence[NoeudTableau]:
        """Les deux fils, la branche choisie en tete, chacun minore par sa penalite."""
        if evaluation.solution is None:
            msg = "noeud sans tableau optimal : il n'y a rien a separer"
            raise ValueError(msg)
        optimal = evaluation.solution.noeud
        choix = choisir_la_separation(optimal.tableau, self.entieres, self.regle)
        variable, branche = choix
        valeur = optimal.tableau.valeur
        fils = tuple(
            replace(noeud, minorant=self.arrondir(valeur + penalite))
            for noeud, penalite in zip(
                separer_le_tableau(optimal, variable),
                penalites(optimal.tableau, variable),
                strict=True,
            )
        )
        return fils if branche is Branche.BASSE else fils[::-1]


def resoudre_par_reoptimisation(
    depart: TableauSimplexe,
    entieres: frozenset[int],
    regle: Regle,
    *,
    couts_entiers: bool = False,
) -> Exploration[NoeudTableau, OptimumDuNoeud]:
    """L'optimum entier, chaque noeud reoptimise par l'algorithme dual."""
    regles = Reoptimisation(entieres, regle, couts_entiers)
    procedure = Procedure(regles.evaluer, regles.separer, lambda n: n.minorant)
    return explorer(NoeudTableau(depart), procedure, Cheminement.PROFONDEUR_D_ABORD)
