"""Rendre triangulaire une matrice qui n'est pas diagonalisable.

Quand une valeur propre a moins de vecteurs propres independants que sa
multiplicite algebrique, la matrice ne se diagonalise pas. Elle reste
**triangularisable** des que son polynome caracteristique a toutes ses
racines reelles : il existe `P` telle que `P^-1 A P` soit triangulaire
superieure, les valeurs propres sur la diagonale.

Le procede : prendre les vecteurs propres, les **completer en une base**
par des vecteurs de la base canonique, et changer de base. La matrice
obtenue porte les valeurs propres en tete et un bloc plus petit en bas a
droite, sur lequel on recommence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


def semblable_par(matrice: sp.Matrix, passage: sp.Matrix) -> sp.Matrix:
    """`P^-1 A P`, la matrice dans la base que portent les colonnes de `P`."""
    return sp.simplify(passage.inv() * matrice * passage)


def est_triangulaire_superieure(matrice: sp.Matrix) -> bool:
    """Vrai si la matrice est carree et nulle sous sa diagonale (II.6)."""
    return bool(matrice.is_square and matrice.is_upper)


def est_triangularisable(matrice: sp.Matrix) -> bool:
    """Vrai si les valeurs propres reelles, avec multiplicite, sont au nombre de `n`."""
    reelles = [m for valeur, m in matrice.eigenvals().items() if valeur.is_real]
    return bool(sum(reelles) == matrice.rows)


def completer_en_base(vecteurs: Sequence[sp.Matrix]) -> sp.Matrix:
    """Les vecteurs donnes, prolonges par des vecteurs canoniques jusqu'a une base."""
    base = sp.Matrix.hstack(*vecteurs)
    if base.rank() < len(vecteurs):
        msg = "famille liee : elle ne se complete pas en une base"
        raise ValueError(msg)
    for k in range(base.rows):
        candidate = sp.Matrix.hstack(base, sp.eye(base.rows)[:, k])
        if candidate.rank() > base.rank():
            base = candidate
    return base


def triangulariser(matrice: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """`(P, T)` avec `P^-1 A P = T` triangulaire superieure ; leve si impossible."""
    if not est_triangularisable(matrice):
        msg = "matrice non triangularisable sur les reels : valeurs propres complexes"
        raise ValueError(msg)
    propres = [v for _, _, vecteurs in matrice.eigenvects() for v in vecteurs]
    passage = completer_en_base(propres)
    reduite = semblable_par(matrice, passage)
    acquis = len(propres)
    if acquis == matrice.rows:
        return passage, reduite
    interieur, _ = triangulariser(reduite[acquis:, acquis:])
    suite = sp.diag(sp.eye(acquis), interieur)
    return passage * suite, semblable_par(reduite, suite)
