"""Chapitre II.3 : le changement de base, par les quatre regles du cours.

Passer d'un sommet a un sommet adjacent, c'est changer un seul vecteur de
la base : `x_k`, hors base, y entre ; `x_l`, en base, en sort. Le
coefficient `a_lk` est le **pivot**. Le nouveau tableau se calcule a
partir de l'ancien (page 54) :

* remplacer le pivot par son inverse ;
* diviser les autres elements de la ligne du pivot par le pivot ;
* diviser les autres elements de la colonne du pivot par le pivot, et
  changer leur signe ;
* pour tous les autres, la **regle du rectangle** :
  `a_ij - a_ik a_lj / a_lk`.

La ligne de `z` et la colonne des termes independants se transforment
comme les autres. Dans le tableau, la variable sortie prend la place de
la variable entree.
"""

from __future__ import annotations

from polytech.optimisation.domain.tableau import TableauSimplexe


def _remplacer(indices: tuple[int, ...], ancien: int, nouveau: int) -> tuple[int, ...]:
    return tuple(nouveau if indice == ancien else indice for indice in indices)


def changer_de_base(
    tableau: TableauSimplexe,
    sortante: int,
    entrante: int,
) -> TableauSimplexe:
    """Le tableau de la base ou `x_entrante` remplace `x_sortante`."""
    if sortante not in tableau.base or entrante not in tableau.hors_base:
        msg = (
            f"x_{sortante} n'est pas en base ou x_{entrante} n'est pas hors "
            f"base : base {tableau.base}, hors base {tableau.hors_base}"
        )
        raise ValueError(msg)
    ligne = tableau.base.index(sortante) + 1
    colonne = tableau.hors_base.index(entrante) + 1
    ancien = tableau.coefficients
    pivot = ancien[ligne, colonne]
    if pivot == 0:
        msg = f"pivot nul en (x_{sortante}, x_{entrante}) : ce ne serait pas une base"
        raise ValueError(msg)
    nouveau = ancien.copy()
    for i in range(ancien.rows):
        for j in range(ancien.cols):
            if i == ligne and j == colonne:
                nouveau[i, j] = 1 / pivot
            elif i == ligne:
                nouveau[i, j] = ancien[i, j] / pivot
            elif j == colonne:
                nouveau[i, j] = -ancien[i, j] / pivot
            else:
                nouveau[i, j] = (
                    ancien[i, j] - ancien[i, colonne] * ancien[ligne, j] / pivot
                )
    return TableauSimplexe(
        base=_remplacer(tableau.base, sortante, entrante),
        hors_base=_remplacer(tableau.hors_base, entrante, sortante),
        coefficients=nouveau,
    )
