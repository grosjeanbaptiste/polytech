"""Recueil d'algebre, chapitre IX, exercices 1 a 16 : normes, bases, projections.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 141 a 143.
Le produit scalaire est l'usuel, `somme(x_i conj(y_i))` dans `C^n`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre import orthogonalisation as orth
from polytech.maths.domain.algebre import produits_scalaires as ps

M = sp.Matrix
I: Final = sp.I  # noqa: E741  # l'unite imaginaire, que le syllabus note i
USUEL = ps.usuel


def col(*composantes: object) -> sp.Matrix:
    """Le vecteur colonne de ces composantes."""
    return M(list(composantes))


FAMILLES: Final = {
    "3d": (col(1, 1, 0, 1), col(1, -2, 0, 0), col(1, 0, -1, 2)),
    "4a": (col(1, I, 0), col(1, 1, 1)),
    "4b": (col(1, -1, -I), col(I, 1, 2)),
    "5a": (col(1, 1, -1), col(1, 0, 1)),
    "5b": (col(2, 1, 1), col(1, 3, -1)),
    "6": (col(1, 2, 1, 0), col(1, 2, 3, 1)),
    "7a": (col(1, 1, 0, 0), col(1, -1, 1, 1), col(-1, 0, 2, 1)),
    "13": (col(0, 0, I, -I), col(-1, I, -I, I)),
}
"""Cle : la famille a orthonormaliser, dans l'ordre de l'enonce."""
EQUATIONS_9: Final = M([[1, -I, 0, 0], [0, 0, 1, 1]])
GENERATEURS_10: Final = M([[1, 1, -2, 3, 4, 5], [0, 0, 1, 1, 0, 7]])
GENERATEURS_12: Final = M([[3, 2, I], [I, 0, -3]])
DROITE_15: Final = M([[1, 1, 1], [4, 1, -1]])
NORMALE_16: Final = col(2, -3, 6)


def _orthogonale(cle: str) -> tuple[sp.Matrix, ...]:
    return orth.gram_schmidt(FAMILLES[cle], USUEL)


def orthogonales() -> dict[str, tuple[sp.Matrix, ...]]:
    """Chaque famille apres Gram-Schmidt, avant d'etre normee."""
    return {cle: _orthogonale(cle) for cle in FAMILLES}


def orthonormales() -> dict[str, tuple[sp.Matrix, ...]]:
    """Chaque famille, orthogonalisee par Gram-Schmidt puis normee."""
    return {cle: orth.normaliser(_orthogonale(cle), USUEL) for cle in FAMILLES}


def exercice_3() -> dict[str, bool]:
    """Les trois vecteurs : une base de `W`, orthogonale, unitaires."""
    famille = FAMILLES["3d"]
    return {
        "a": M.hstack(*famille).rank() == len(famille),
        "b": ps.est_orthogonale(famille, USUEL),
        "c": all(ps.norme(v, USUEL) == 1 for v in famille),
    }


def exercice_9() -> tuple[sp.Matrix, tuple[sp.Matrix, ...]]:
    """La projection imposee, et une base de `W^perp` : la somme les donne tous."""
    return col(1, -I, 1, -1), tuple(EQUATIONS_9.H.col(j) for j in range(2))


def exercice_10() -> tuple[sp.Matrix, ...]:
    """Une base de l'orthogonal des deux vecteurs de `R^6`."""
    return tuple(GENERATEURS_10.nullspace())


def exercice_11(dimension: int) -> tuple[int, tuple[sp.Matrix, ...]]:
    """La dimension de `W = {somme des x_i = 0}`, et une base de `W^perp`."""
    if dimension < 1:
        msg = f"dimension non strictement positive : {dimension}"
        raise ValueError(msg)
    equation = sp.ones(1, dimension)
    return len(equation.nullspace()), (equation.T,)


def exercice_12() -> tuple[sp.Matrix, ...]:
    """Une base de l'orthogonal dans `C^3` : `<X | g> = 0` pour chaque generateur."""
    return tuple(GENERATEURS_12.conjugate().nullspace())


def exercice_14() -> sp.Matrix:
    """La projection orthogonale de `X` sur le sous-espace de l'exercice 13."""
    return orth.projection(col(3, I, 1, -1), _orthogonale("13"), USUEL)


def nombres() -> dict[str, sp.Expr]:
    """Les normes, distances et produits scalaires des exercices 1 a 16."""
    x1, x2 = col(1, 2, 3), col(-1, 3, 4)
    x, y = col(3, 2, I), col(I, I, -3)
    direction = DROITE_15.nullspace()
    point = col(1, 5, 0)
    return {
        "1NormeUn": ps.norme(x1, USUEL),
        "1NormeDeux": ps.norme(x2, USUEL),
        "1Distance": ps.norme(x1 - x2, USUEL),
        "1Produit": USUEL(x1, x2),
        "2Produit": USUEL(x, y),
        "2Norme": ps.norme(x, USUEL),
        "7b": orth.distance_au_sous_espace(col(1, 1, 1, -3), _orthogonale("7a"), USUEL),
        "8": orth.distance_au_sous_espace(col(I, I, I), _orthogonale("4a"), USUEL),
        "10a": sp.Integer(len(exercice_10())),
        "14b": ps.norme(col(3, I, 1, -1) - exercice_14(), USUEL),
        "15": orth.distance_au_sous_espace(col(1, 2, 3), direction, USUEL),
        "16": sp.Abs(USUEL(point, NORMALE_16)) / ps.norme(NORMALE_16, USUEL),
    }
