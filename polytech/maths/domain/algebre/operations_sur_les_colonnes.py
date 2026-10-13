"""Les operations elementaires sur les colonnes d'un determinant.

Le syllabus d'algebre (V.1, page 55) les autorise « sur les lignes et/ou
sur les colonnes », et enonce ses trois proprietes pour une « rangee ».
C'est ainsi que l'exemple X.2.4 (page 157) factorise son polynome
caracteristique : `c1 <- c1 + c2`, puis `c1 <- c1 + c3`.

Une operation sur les colonnes est la meme operation sur les lignes de la
transposee, et `det(A) = det(A^t)` : le facteur est donc celui des lignes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.maths.domain.algebre import operations_elementaires as lignes
from polytech.maths.domain.algebre.operations_elementaires import Transformee

if TYPE_CHECKING:
    import sympy as sp


def _transposee(transformee: Transformee) -> Transformee:
    return Transformee(transformee.matrice.T, transformee.facteur)


def permuter(matrice: sp.Matrix, premiere: int, seconde: int) -> Transformee:
    """Propriete 1 : permuter deux colonnes change le signe du determinant."""
    return _transposee(lignes.permuter(matrice.T, premiere, seconde))


def multiplier(matrice: sp.Matrix, colonne: int, alpha: sp.Expr) -> Transformee:
    """Propriete 2 : le determinant est multiplie par `alpha`."""
    return _transposee(lignes.multiplier(matrice.T, colonne, alpha))


def combiner(
    matrice: sp.Matrix, colonne: int, beta: sp.Expr, autre: int
) -> Transformee:
    """Propriete 3 : `c_i -> c_i + beta c_k` laisse le determinant inchange."""
    if colonne == autre:
        msg = "une colonne ne se combine pas avec elle-meme : ce serait la multiplier"
        raise ValueError(msg)
    return _transposee(lignes.combiner(matrice.T, colonne, beta, autre))
