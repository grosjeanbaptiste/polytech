"""Recueil d'algebre, chapitre VIII, exercices 14 a 16, 18, 19, 22, 28.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, section VIII.7
(pages 112 a 115) : la matrice associee a une application dans des bases
donnees. Une matrice 2 x 2 se lit ligne apres ligne.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.espaces_de_fonctions import matrice_d_un_operateur

if TYPE_CHECKING:
    from collections.abc import Callable

M = sp.Matrix
LETTRES: Final = sp.symbols("a b c d")
T: Final = sp.Symbol("t")
FIXE: Final = M(2, 2, LETTRES)
"""La matrice `M` de l'exercice 19."""
_REGLE = ApplicationLineaire.depuis_la_regle
BASES_18: Final = {
    "a": (sp.exp(T), sp.exp(2 * T)),
    "b": (sp.Integer(1), T),
    "c": (sp.exp(T), T * sp.exp(T)),
    "d": (sp.Integer(1), T, T**2),
    "e": (sp.sin(T), sp.cos(T)),
}
BASE_22: Final = tuple(T**k * sp.exp(T) for k in range(4))
F_28: Final = {
    "f1": _REGLE(lambda x, y: (x, 3 * y, y - x, x + y), 2).matrice,
    "f2": _REGLE(lambda x, y: (x + y, 0, 0, 0), 2).matrice,
    "f3": _REGLE(lambda a, b, c, d: (a + b, c + d), 4).matrice,
    "f4": _REGLE(lambda _a, _b, c, _d: (0, c), 4).matrice,
}


def _sur_les_matrices(regle: Callable[[sp.Matrix], sp.Matrix]) -> sp.Matrix:
    return _REGLE(lambda *termes: list(regle(M(2, 2, termes))), 4).matrice


def _derivation(base: tuple[sp.Expr, ...]) -> sp.Matrix:
    return matrice_d_un_operateur(lambda f: sp.diff(f, T), base, T)


def _du_16a(m: sp.Matrix) -> sp.Matrix:
    return M(
        [[m[0, 0] + m[0, 1], m[0, 1] + m[1, 0]], [m[1, 0] + m[1, 1], m[1, 1] + m[0, 0]]]
    )


def _du_28() -> dict[str, sp.Matrix]:
    f1, f2, f3, f4 = (F_28[k] for k in ("f1", "f2", "f3", "f4"))
    composees = {
        "somme": f1 + f2,
        "double": 2 * f2,
        "difference": 3 * f4 - f3,
        "f4f1": f4 * f1,
        "f2f3": f2 * f3,
        "f3f2f4": f3 * f2 * f4,
    }
    return {f"28{k}": m for k, m in (F_28 | composees).items()}


def toutes() -> dict[str, sp.Matrix]:
    """Chaque matrice que ces exercices demandent, par sa cle."""
    translation = matrice_d_un_operateur(lambda p: p.subs(T, T + 1), BASES_18["d"], T)
    derivation = _derivation(BASE_22)
    matrices = {
        "14a": M([[2, 1], [1, 0]]) * M([3, -1]),
        "14b": M([[2, 1], [1, 0], [-1, 4]]) * M([5, 1]),
        "15a": _REGLE(lambda a, b, _c, _d: (a, b), 4).matrice,
        "16a": _sur_les_matrices(_du_16a),
        "16b": translation,
        "19f1": _sur_les_matrices(lambda m: FIXE * m),
        "19f2": _sur_les_matrices(lambda m: m * FIXE),
        "19f3": _sur_les_matrices(lambda m: FIXE * m - m * FIXE),
        "22": derivation,
        "22inverse": derivation.inv(),
    }
    matrices |= {f"18{k}": _derivation(base) for k, base in BASES_18.items()}
    return matrices | _du_28()


def homothetie(ordre: int) -> sp.Matrix:
    """15 b) : la matrice de `X -> 7 X` sur `R^n`."""
    return _REGLE(lambda *x: [7 * c for c in x], ordre).matrice


def primitive_du_22() -> sp.Expr:
    """La primitive de `t^3 e^t` : l'inverse de la derivation, applique a la base."""
    coordonnees = toutes()["22inverse"] * M([0, 0, 0, 1])
    return sp.Add(*[c * f for c, f in zip(coordonnees, BASE_22, strict=True)])
