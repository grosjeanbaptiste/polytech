"""Recueil d'algebre, chapitre V : calculs et proprietes des determinants.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 8 de
la section V.6 (pages 60 a 62). Les matrices d'ordre `n` -- 1 f), 3, 5, 6
-- se calculent a un ordre donne ; le corrige raisonne pour tout `n`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import sympy as sp

from polytech.maths.application.formatage import coordonnees
from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.operations_elementaires import par_echelonnement

M = sp.Matrix
LETTRES: Final = sp.symbols("a b c d")
A, B, C, D = LETTRES
INCONNUE: Final = sp.Symbol("x")
LAMBDA: Final = sp.Symbol("lambda")
MATRICES_1: Final = {
    "a": M([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, 1, 1]]),
    "b": M([[1 + A, 1, 1, 1], [1, 1 + B, 1, 1], [1, 1, 1 + C, 1], [1, 1, 1, 1 + D]]),
    "c": M([[1, 2, 3, 4], [5, 6, 7, 8], [8, 4, 6, 5], [4, 4, 2, -2]]),
    "d": M(
        [
            [1, 0, -3, 0, 7, 2],
            [11, -3, 2, 1, 10, -2],
            [0, 0, 1, 0, 6, 0],
            [5, 1, 4, 0, -9, 6],
            [0, 0, 0, 0, 2, 0],
            [-3, 0, -10, 0, 4, 1],
        ]
    ),
    "e": M([[1, A, B * C], [1, B, A * C], [1, C, A * B]]),
}
MATRICE_2: Final = M([[A, 1, -1, 1], [-1, A, -1, -1], [1, 1, A, -1], [-1, 1, 1, A]])
MATRICE_7: Final = M([[LAMBDA - 1, 3, -3], [-3, LAMBDA + 5, -3], [-6, 6, LAMBDA - 4]])
_X2 = INCONNUE**2
MATRICE_8: Final = M(
    [[2, 2, 3, 4], [2, 2 - _X2, 3, 4], [2, 3, 1, 5], [2, 3, 1, 9 - _X2]]
)


def _exiger_un_ordre(ordre: int) -> None:
    if ordre < 1:
        msg = f"ordre non strictement positif : {ordre}"
        raise ValueError(msg)


def exercice_1() -> dict[str, sp.Expr]:
    """Les cinq determinants a), ... e) ; les numeriques par echelonnement."""
    return {
        lettre: sp.factor(par_echelonnement(m) if not m.free_symbols else m.det())
        for lettre, m in MATRICES_1.items()
    }


def tridiagonale(ordre: int) -> sp.Expr:
    """1 f) : des 2 sur la diagonale, des 1 de part et d'autre."""
    _exiger_un_ordre(ordre)
    return M(ordre, ordre, lambda i, j: {0: 2, 1: 1}.get(abs(i - j), 0)).det()


def exercice_2() -> tuple[sp.Matrix, sp.Expr]:
    """`A A^t`, puis le determinant de `A`."""
    return sp.simplify(MATRICE_2 * MATRICE_2.T), sp.factor(MATRICE_2.det())


def vandermonde(ordre: int) -> tuple[sp.Expr, sp.Expr]:
    """Le determinant de Vandermonde d'ordre `n`, et le produit des `a_i - a_j`."""
    _exiger_un_ordre(ordre)
    points = sp.symbols(f"a1:{ordre + 1}")
    determinant = M(ordre, ordre, lambda i, j: points[i] ** j).det()
    produit = sp.Mul(*[points[i] - points[j] for i in range(ordre) for j in range(i)])
    return sp.factor(determinant), produit


def matrices_du_4() -> tuple[sp.Matrix, sp.Matrix]:
    """Celle de gauche, de colonnes `B + C`, `C + A`, `A + B`, et celle de droite."""
    droite = M(3, 3, sp.symbols("a b c a1 b1 c1 a2 b2 c2"))
    gauche = M(3, 3, lambda i, j: droite[i, (j + 1) % 3] + droite[i, (j + 2) % 3])
    return gauche, droite


def exercice_4() -> tuple[sp.Expr, sp.Expr]:
    """Les deux determinants de l'egalite a demontrer."""
    gauche, droite = matrices_du_4()
    return gauche.det(), droite.det()


def exercice_5(ordre: int) -> sp.Expr:
    """Le determinant des entiers de 1 a `n^2` ranges ligne par ligne."""
    _exiger_un_ordre(ordre)
    return M(ordre, ordre, lambda i, j: i * ordre + j + 1).det()


def exercice_6(matrice: sp.Matrix) -> tuple[sp.Expr, sp.Expr]:
    """Le determinant de la matrice des cofacteurs, et celui de la matrice."""
    return Determinant(matrice).comatrice().det(), matrice.det()


def exercice_7() -> tuple[sp.Expr, dict[sp.Expr, int]]:
    """Le determinant factorise, et ses racines avec leur ordre."""
    determinant = sp.factor(MATRICE_7.det())
    return determinant, sp.roots(determinant, LAMBDA)


def exercice_8() -> tuple[sp.Expr, dict[sp.Expr, int]]:
    """Le determinant factorise, et ses racines avec leur ordre."""
    determinant = sp.factor(MATRICE_8.det())
    return determinant, sp.roots(determinant, INCONNUE)


def courbe_du_7() -> str:
    """Le determinant de l'exercice 7 en fonction de `lambda`."""
    abscisses = np.linspace(-4.0, 5.0, 46)
    ordonnees = sp.lambdify(LAMBDA, exercice_7()[0], "numpy")(abscisses)
    return coordonnees(abscisses, ordonnees, 2)
