"""Chapitre I.3.1 a I.3.3 : les rappels d'algebre lineaire du cours.

**Espace vectoriel.** Des vecteurs sont lineairement independants si
leur seule combinaison lineaire nulle est la triviale. Une base de
`R^n` en compte `n`, et tout vecteur s'y ecrit de maniere **unique**.
Le lemme sur lequel repose le simplexe : substituer un vecteur `x` au
vecteur `x_j` de la base donne une nouvelle base si et seulement si la
coordonnee `lambda_j` de `x` est non nulle.

**Rang.** Le nombre maximal de lignes independantes egale celui des
colonnes : c'est le rang `r(A)`. Une matrice carree est **reguliere** si
son rang est son ordre ; elle est alors inversible.

**Systemes d'equations** `A x = b`, a `n` inconnues :
`r(A) < r(A, b)`, impossible ; `r(A) = r(A, b) = n`, solution unique ;
`r(A) = r(A, b) < n`, infinite de solutions. Multiplier par une matrice
reguliere `B` ne change pas l'ensemble des solutions.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


class NatureDuSysteme(Enum):
    """Les trois situations d'un systeme lineaire (I.3.3)."""

    IMPOSSIBLE = "aucune solution"
    SOLUTION_UNIQUE = "une solution unique"
    INFINITE_DE_SOLUTIONS = "une infinite de solutions"


def rang(matrice: sp.Matrix) -> int:
    """`r(A)` : le nombre maximal de lignes -- ou de colonnes -- independantes."""
    return int(matrice.rank())


def sont_lineairement_independants(vecteurs: Sequence[sp.Matrix]) -> bool:
    """Vrai si la seule combinaison lineaire nulle est la triviale."""
    return rang(sp.Matrix.hstack(*vecteurs)) == len(vecteurs)


def est_reguliere(matrice: sp.Matrix) -> bool:
    """Une matrice carree dont le rang est l'ordre."""
    return bool(matrice.is_square) and rang(matrice) == matrice.rows


def coordonnees_dans_la_base(
    vecteur: sp.Matrix, base: Sequence[sp.Matrix]
) -> sp.Matrix:
    """Les `lambda_j` uniques tels que `vecteur = somme(lambda_j base_j)`."""
    matrice = sp.Matrix.hstack(*base)
    if not est_reguliere(matrice):
        msg = f"ces {len(base)} vecteurs ne forment pas une base de R^{matrice.rows}"
        raise ValueError(msg)
    return matrice.solve(vecteur)


def substituer_dans_la_base(
    base: Sequence[sp.Matrix],
    rang_du_sortant: int,
    entrant: sp.Matrix,
) -> list[sp.Matrix]:
    """La base ou `entrant` remplace le vecteur de rang donne ; leve si `lambda = 0`."""
    if coordonnees_dans_la_base(entrant, base)[rang_du_sortant] == 0:
        msg = f"coordonnee nulle sur le vecteur {rang_du_sortant} : pas une base"
        raise ValueError(msg)
    nouvelle = list(base)
    nouvelle[rang_du_sortant] = entrant
    return nouvelle


def nature_du_systeme(matrice: sp.Matrix, second_membre: sp.Matrix) -> NatureDuSysteme:
    """Impossible, solution unique, ou infinite de solutions."""
    if rang(matrice) < rang(matrice.row_join(second_membre)):
        return NatureDuSysteme.IMPOSSIBLE
    if rang(matrice) == matrice.cols:
        return NatureDuSysteme.SOLUTION_UNIQUE
    return NatureDuSysteme.INFINITE_DE_SOLUTIONS


def systeme_equivalent(
    matrice: sp.Matrix,
    second_membre: sp.Matrix,
    reguliere: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix]:
    """`B A x = B b`, qui a les memes solutions que `A x = b`."""
    if not est_reguliere(reguliere):
        msg = "matrice non reguliere : le systeme produit perdrait des equations"
        raise ValueError(msg)
    return reguliere * matrice, reguliere * second_membre
