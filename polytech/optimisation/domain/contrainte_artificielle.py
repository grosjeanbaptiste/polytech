"""Chapitre V.3 : la phase I de l'algorithme dual, par la contrainte artificielle.

Quand la base disponible n'est pas duale admissible, on borne ce qui la
rend telle : avec `K = {j | a_0j > 0}` (en minimisation), on ajoute au
probleme la **contrainte artificielle**

    v + somme(x_j, j dans K) = M,     v >= 0,

`M` aussi grand que l'on veut. Un changement de base **impose** -- `v`
sort, la variable de `K` au plus grand `a_0j` entre -- rend la base duale
admissible, et l'algorithme dual s'applique. Trois conclusions :

1. le probleme augmente n'a pas de solution admissible : `P` non plus ;
2. optimum avec `v` hors base : la contrainte artificielle est serree,
   une variable au moins depend de `M`. Si `z` en depend, `P` n'est pas
   borne ; sinon c'est un rayon de solutions optimales ;
3. optimum avec `v` en base : c'est une solution optimale de `P`.

Des que `v` rentre en base elle n'en sort plus : sa ligne est supprimee.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.optimisation.domain.algorithme import ITERATIONS_MAXIMALES, Issue, Regle
from polytech.optimisation.domain.algorithme_dual import (
    phase_deux_duale,
    pivoter_en_dual,
)
from polytech.optimisation.domain.artificielles import sans_ligne
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres_duaux import (
    est_primal_admissible,
    ligne_sans_solution_admissible,
)
from polytech.optimisation.domain.grand_m import GRAND_M, ordre
from polytech.optimisation.domain.pivot import changer_de_base
from polytech.optimisation.domain.tableau import TableauSimplexe


@dataclass(frozen=True, slots=True)
class ResolutionDuale:
    """Les tableaux, l'arret, et l'indice de `v` si une contrainte a ete ajoutee."""

    tableaux: tuple[TableauSimplexe, ...]
    issue: Issue
    artificielle: int | None = None


def _violations(tableau: TableauSimplexe, sens: Optimisation) -> dict[int, sp.Expr]:
    signe = 1 if sens is Optimisation.MIN else -1
    coefficients = zip(tableau.hors_base, tableau.coefficients[0, 1:], strict=True)
    return {j: signe * a for j, a in coefficients if signe * a > 0}


def indices_a_borner(tableau: TableauSimplexe, sens: Optimisation) -> tuple[int, ...]:
    """`K` : les indices hors base qui violent la condition duale."""
    return tuple(_violations(tableau, sens))


def probleme_augmente_dual(
    tableau: TableauSimplexe,
    sens: Optimisation,
) -> TableauSimplexe:
    """Le tableau, plus la ligne `v + somme(x_j, j dans K) = M`."""
    borner = indices_a_borner(tableau, sens)
    if not borner:
        msg = f"base {tableau.base} deja duale admissible : aucune contrainte a ajouter"
        raise ValueError(msg)
    artificielle = max((*tableau.base, *tableau.hors_base)) + 1
    ligne = sp.Matrix([[GRAND_M, *(int(j in borner) for j in tableau.hors_base)]])
    return TableauSimplexe(
        (*tableau.base, artificielle),
        tableau.hors_base,
        tableau.coefficients.col_join(ligne),
    )


def _conclusion(tableau: TableauSimplexe) -> Issue:
    """Cas 2 et 3 : `z` ne peut dependre de `M` que si `v` est hors base.

    Quand `v` est en base, elle est « la seule a dependre de la constante
    `M` » (page 120), et sa ligne a ete supprimee : lire `z` suffit.
    """
    depend_de_m = ordre(tableau.valeur)[0] != 0
    return Issue.NON_BORNEE if depend_de_m else Issue.OPTIMALE


def methode_de_la_contrainte_artificielle(
    depart: TableauSimplexe,
    sens: Optimisation,
    regle: Regle = Regle.PLUS_FORT_COEFFICIENT,
    iterations_maximales: int = ITERATIONS_MAXIMALES,
) -> ResolutionDuale:
    """Phase I puis phase II de l'algorithme dual, depuis une base quelconque."""
    violations = _violations(depart, sens)
    if not violations:
        directe = phase_deux_duale(depart, sens, regle)
        return ResolutionDuale(directe.tableaux, directe.issue)
    augmente = probleme_augmente_dual(depart, sens)
    v = augmente.base[-1]
    entrante = max(violations, key=lambda j: violations[j])
    tableaux = [augmente, changer_de_base(augmente, v, entrante)]
    for _ in range(iterations_maximales):
        courant = tableaux[-1]
        if v in courant.base:
            courant = sans_ligne(courant, v)
            tableaux.append(courant)
        if est_primal_admissible(courant):
            return ResolutionDuale(tuple(tableaux), _conclusion(courant), v)
        if ligne_sans_solution_admissible(courant) is not None:
            return ResolutionDuale(tuple(tableaux), Issue.IMPOSSIBLE, v)
        tableaux.append(pivoter_en_dual(courant, regle))
    msg = "aucun arret de la methode de la contrainte artificielle : cyclage ?"
    raise RuntimeError(msg)


def origine_du_rayon(tableau: TableauSimplexe, variables: int) -> sp.Matrix:
    """Le point de depart du rayon : `M` a sa plus petite valeur gardant `x >= 0`."""
    seuils = [sp.Integer(0)]
    for valeur in tableau.coefficients[1:, 0]:
        pente, constante = ordre(valeur)
        if pente > 0:
            seuils.append(-constante / pente)
    return tableau.solution(variables).subs(GRAND_M, max(seuils))
