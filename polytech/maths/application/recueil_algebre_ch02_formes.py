"""Recueil d'algebre, chapitre II : echelonnement simple et inverses.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 9 et 10
(sections II.7.4 et II.8.3). Les matrices 10 e) et 10 f) sont d'ordre `n` :
elles se calculent ici a un ordre donne, et le corrige raisonne pour tout `n`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.echelonnement import (
    echelonner,
    echelonner_simplement,
    inverse_par_la_matrice_compagnon,
)

M = sp.Matrix
MATRICES_9: Final = {
    "A": M([[1, -2, 3, -1], [2, -1, 2, 2], [-3, -5, 1, -5], [3, 1, 2, 3]]),
    "B": M(
        [[1, 2, 1, 2, -3], [3, 6, 4, -1, -1], [4, 8, 5, 1, -1], [-2, -4, -3, 3, -5]]
    ),
    "C": M([[6, 3, -4], [-4, 1, -6], [1, 2, -5]]),
    "D": M(
        [[1, 3, 1, -2, -3], [1, 4, 3, -1, -4], [2, 3, -4, -7, -3], [3, 8, 1, -7, -8]]
    ),
}
MATRICES_10: Final = {
    "a": M([[1, 2, -1], [0, 3, 0], [-2, 1, 0]]),
    "b": M([[1, 2, 3], [4, 5, 6], [7, 8, 10]]),
    "c": M([[1, -2, 3, -1], [2, -1, 2, 2], [-3, -5, 1, -5], [3, 1, 2, 4]]),
    "d": M([[1, 1, 1, 1], [1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1]]),
}


def exercice_9() -> dict[str, sp.Matrix]:
    """La forme echelonnee simple de chacune des quatre matrices."""
    return {nom: echelonner_simplement(m) for nom, m in MATRICES_9.items()}


def echelonnees_du_9() -> dict[str, sp.Matrix]:
    """La forme echelonnee, etape intermediaire que le corrige montre."""
    return {nom: echelonner(m).matrice for nom, m in MATRICES_9.items()}


def journaux_du_9() -> dict[str, tuple[str, ...]]:
    """Les operations sur les lignes qui menent a chaque forme echelonnee."""
    return {nom: echelonner(m).operations for nom, m in MATRICES_9.items()}


def exercice_10() -> dict[str, sp.Matrix]:
    """L'inverse des quatre matrices numeriques, par la matrice-compagnon."""
    return {k: inverse_par_la_matrice_compagnon(m) for k, m in MATRICES_10.items()}


def _exiger_un_ordre(ordre: int) -> None:
    if ordre < 1:
        msg = f"ordre non strictement positif : {ordre}"
        raise ValueError(msg)


def inverse_de_la_bidiagonale(ordre: int) -> sp.Matrix:
    """10 e) : des 1 sur la diagonale, des -1 juste au-dessus ; son inverse."""
    _exiger_un_ordre(ordre)
    bidiagonale = sp.eye(ordre) - M(ordre, ordre, lambda i, j: int(j == i + 1))
    return inverse_par_la_matrice_compagnon(bidiagonale)


def antidiagonale_et_son_inverse(ordre: int) -> tuple[sp.Matrix, sp.Matrix]:
    """10 f) : `lambda_k` en ligne `k`, colonne `n + 1 - k`, et son inverse."""
    _exiger_un_ordre(ordre)
    lambdas = sp.symbols(f"lambda1:{ordre + 1}")
    matrice = M(ordre, ordre, lambda i, j: lambdas[i] if i + j == ordre - 1 else 0)
    return matrice, matrice.inv()
