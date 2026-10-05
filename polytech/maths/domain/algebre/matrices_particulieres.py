"""Trace, matrices particulieres, et les matrices qui commutent avec une autre.

Le syllabus d'algebre nomme cinq matrices carrees remarquables (II.4) :
symetrique `A = A^t`, antisymetrique `A = -A^t`, diagonale, triangulaire
superieure, triangulaire inferieure ; et, a l'exercice 7, la matrice
orthogonale `A A^t = A^t A = I`. Toute matrice carree est la somme d'une
symetrique `(A + A^t)/2` et d'une antisymetrique `(A - A^t)/2`.

Chacune de ces definitions dit « matrice carree » : une matrice
rectangulaire n'en verifie aucune, et la trace, elle, la refuse.
"""

from __future__ import annotations

import sympy as sp


def _est_carree(matrice: sp.Matrix) -> bool:
    return bool(matrice.rows == matrice.cols)


def _est_nulle(matrice: sp.Matrix) -> bool:
    return bool(sp.simplify(matrice) == sp.zeros(matrice.rows, matrice.cols))


def trace(matrice: sp.Matrix) -> sp.Expr:
    """La somme des elements diagonaux."""
    if not _est_carree(matrice):
        msg = "trace d'une matrice non carree"
        raise ValueError(msg)
    return sum((matrice[i, i] for i in range(matrice.rows)), sp.Integer(0))


def est_symetrique(matrice: sp.Matrix) -> bool:
    """`A = A^t`."""
    return _est_carree(matrice) and _est_nulle(matrice - matrice.T)


def est_antisymetrique(matrice: sp.Matrix) -> bool:
    """`A = -A^t`, donc une diagonale nulle."""
    return _est_carree(matrice) and _est_nulle(matrice + matrice.T)


def est_triangulaire_inferieure(matrice: sp.Matrix) -> bool:
    """Carree, et nulle au-dessus de la diagonale."""
    return _est_carree(matrice) and all(
        matrice[i, j] == 0
        for i in range(matrice.rows)
        for j in range(i + 1, matrice.cols)
    )


def est_diagonale(matrice: sp.Matrix) -> bool:
    """Carree, et nulle hors de la diagonale."""
    return est_triangulaire_inferieure(matrice) and est_triangulaire_inferieure(
        matrice.T
    )


def est_une_matrice_orthogonale(matrice: sp.Matrix) -> bool:
    """`A A^t = A^t A = I`."""
    return _est_carree(matrice) and _est_nulle(
        matrice * matrice.T - sp.eye(matrice.rows)
    )


def parties_symetrique_et_antisymetrique(
    matrice: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix]:
    """`A = (A + A^t)/2 + (A - A^t)/2`."""
    return (matrice + matrice.T) / 2, (matrice - matrice.T) / 2


def commutant(matrice: sp.Matrix) -> sp.Matrix:
    """La matrice generale `X` telle que `A X = X A`, en ses parametres libres."""
    ordre = matrice.rows
    inconnues = sp.symbols(f"x1:{ordre * ordre + 1}")
    generale = sp.Matrix(ordre, ordre, inconnues)
    equations = list(matrice * generale - generale * matrice)
    (solution,) = sp.solve(equations, inconnues, dict=True)
    return generale.subs(solution)
