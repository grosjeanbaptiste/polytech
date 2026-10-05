"""Sous-espaces de `K^n` : base, equations, somme, intersection.

Le syllabus d'algebre (VII.3, VII.4) passe d'une description a l'autre :

* un sous-espace **defini par des equations** `A X = 0` a pour base les
  vecteurs obtenus en posant tour a tour chaque parametre a 1 et les
  autres a 0 ;
* un sous-espace **engendre par une famille** de rang `m` est decrit par
  `n - m` equations independantes : les `c` tels que `c . w = 0` pour
  chaque generateur `w`, c'est-a-dire le noyau de la matrice des
  generateurs.

La somme de deux sous-espaces est engendree par la reunion des familles ;
un vecteur de l'intersection s'ecrit dans les deux bases a la fois, et
`dim(U + V) + dim(U inter V) = dim U + dim V`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.familles_de_vecteurs import (
    extraire_une_base,
    matrice_des_lignes,
    rang_de_la_famille,
)
from polytech.maths.domain.algebre.resolution import par_echelonnement_simple
from polytech.maths.domain.algebre.systemes import SystemeLineaire

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.familles_de_vecteurs import Famille, Vecteur


def base_du_noyau(matrice: sp.Matrix) -> tuple[tuple[sp.Expr, ...], ...]:
    """Une base de `{X | A X = 0}` : chaque variable libre a 1, les autres a 0."""
    homogene = SystemeLineaire(matrice, sp.zeros(matrice.rows, 1))
    return par_echelonnement_simple(homogene).directions


def equations(famille: Famille) -> tuple[tuple[sp.Expr, ...], ...]:
    """Les coefficients de `n - m` equations independantes du sous-espace engendre."""
    return base_du_noyau(matrice_des_lignes(famille))


def somme(premiere: Famille, seconde: Famille) -> tuple[Vecteur, ...]:
    """Une base de `U + V`, extraite de la reunion des deux familles."""
    return extraire_une_base((*premiere, *seconde))


def intersection(premiere: Famille, seconde: Famille) -> tuple[Vecteur, ...]:
    """Une base de `U inter V` : les vecteurs qui s'ecrivent dans les deux bases."""
    gauche = matrice_des_lignes(extraire_une_base(premiere)).T
    droite = matrice_des_lignes(extraire_une_base(seconde)).T
    communs = base_du_noyau(gauche.row_join(-droite))
    return tuple(
        tuple(gauche * sp.Matrix(list(coefficients[: gauche.cols])))
        for coefficients in communs
    )


def sont_supplementaires(premiere: Famille, seconde: Famille) -> bool:
    """`K^n = U + V` en somme directe : la somme est tout, l'intersection est nulle."""
    dimension = len(premiere[0])
    complete = len(somme(premiere, seconde)) == dimension
    directe = rang_de_la_famille(premiere) + rang_de_la_famille(seconde) == dimension
    return complete and directe


def engendrent_le_meme_sous_espace(premiere: Famille, seconde: Famille) -> bool:
    """Reunir les deux familles ne releve le rang d'aucune."""
    ensemble = rang_de_la_famille((*premiere, *seconde))
    return rang_de_la_famille(premiere) == ensemble == rang_de_la_famille(seconde)
