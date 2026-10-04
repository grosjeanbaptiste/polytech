"""Chapitre II.5.3 : ce que l'on fait des variables artificielles dans le tableau.

Trois gestes, tous tires des remarques du cours :

* **supprimer la colonne** d'une variable artificielle sortie de la base
  -- « il est certain qu'elle n'y rentrera plus » (remarque 4 beta) ;
* **chasser de la base** une variable artificielle nulle, par un pivot
  non nul de sa ligne : la valeur de `z` n'en est pas modifiee, et le
  signe du pivot n'a ici aucune importance (remarque 2) ;
* **eliminer la ligne** d'une variable artificielle nulle qu'aucun pivot
  ne peut chasser : la contrainte correspondante etait redondante.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.domain.pivot import changer_de_base
from polytech.optimisation.domain.tableau import TableauSimplexe


def sans_colonnes(
    tableau: TableauSimplexe, indices: tuple[int, ...]
) -> TableauSimplexe:
    """Le tableau sans les colonnes hors base de ces variables."""
    gardees = [r for r, j in enumerate(tableau.hors_base, start=1) if j not in indices]
    return TableauSimplexe(
        base=tableau.base,
        hors_base=tuple(tableau.hors_base[r - 1] for r in gardees),
        coefficients=tableau.coefficients[:, [0, *gardees]],
    )


def sans_ligne(tableau: TableauSimplexe, indice: int) -> TableauSimplexe:
    """Le tableau sans la ligne de cette variable de base."""
    ligne = tableau.base.index(indice) + 1
    gardees = [i for i in range(tableau.coefficients.rows) if i != ligne]
    return TableauSimplexe(
        base=tuple(i for i in tableau.base if i != indice),
        hors_base=tableau.hors_base,
        coefficients=tableau.coefficients[gardees, :],
    )


def chasser_les_nulles(
    tableau: TableauSimplexe,
    artificielles: tuple[int, ...],
) -> tuple[TableauSimplexe, int]:
    """Le tableau sans variable artificielle en base, et le nombre de lignes eliminees.

    Suppose que les variables artificielles en base sont toutes nulles.
    """
    eliminees = 0
    for indice in [i for i in tableau.base if i in artificielles]:
        ligne = tableau.base.index(indice) + 1
        pivots = [
            j
            for rang, j in enumerate(tableau.hors_base, start=1)
            if tableau.coefficients[ligne, rang] != 0
        ]
        if pivots:
            echange = changer_de_base(tableau, sortante=indice, entrante=pivots[0])
            tableau = sans_colonnes(echange, artificielles)
        else:
            tableau = sans_ligne(tableau, indice)
            eliminees += 1
    return tableau, eliminees


def somme_des_artificielles(
    tableau: TableauSimplexe,
    artificielles: tuple[int, ...],
) -> sp.Expr:
    """`xi = somme(x_i^a)` : nulle si `P` a une solution, positive sinon."""
    return sp.Add(
        *(
            tableau.coefficients[ligne, 0]
            for ligne, indice in enumerate(tableau.base, start=1)
            if indice in artificielles
        )
    )
