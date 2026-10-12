"""Recueil d'algebre, chapitre VIII, exercices 27 et 29 a 32.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, section VIII.7
(pages 115 et 116) : noyau et supplementaire, projections, reduction a la
forme `(I | 0)`, application retrouvee a partir de sa matrice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.representation import (
    matrice_associee,
    matrice_canonique,
    matrice_de_passage,
)
from polytech.maths.domain.algebre.sous_espaces import base_du_noyau

if TYPE_CHECKING:
    from polytech.maths.application.recueil_algebre_ch07_familles import Base

M = sp.Matrix
F_27: Final = ApplicationLineaire.depuis_la_regle(
    lambda a, b, c, d: (a - b + c + d, a + 2 * c - d, a + b + 3 * c - 3 * d), 4
)
V_29: Final = ((1, 0, 1), (0, 1, 1), (0, 0, 1))
PROJECTION_F: Final = sp.diag(1, 1, 0)
PROJECTION_G: Final = sp.diag(0, 0, 1)
A_30: Final = M([[2, 5, -3], [1, -4, 7]])
V_30: Final = ((1, 1, 1), (1, 1, 0), (1, 0, 0))
V_PRIME_30: Final = ((1, 3), (2, 5))
V_31: Final = ((1, 1, 1), (1, 1, 0), (1, 0, 0))
W_31: Final = ((1, -1, 0), (2, -1, -2), (1, -1, -2))
M_31: Final = M([[2, 1, 0], [-4, -2, -2], [7, 7, 5]])
EQUATIONS_32: Final = M([[1, -1, -1, 0], [1, 0, 0, -1]])
"""Le noyau du 32 : `a0 - a1 - a2 = 0` et `a0 - a3 = 0`."""
IMAGES_32: Final = {(0, 0, 1, 0): (0, 1, 1, 0), (0, 0, 0, 1): (1, 0, 0, 1)}
"""`f(X^2)` et `f(X^3)`, matrices lues ligne apres ligne."""


def canonique(dimension: int) -> tuple[tuple[int, ...], ...]:
    """La base canonique de `K^n`."""
    return tuple(tuple(int(i == j) for j in range(dimension)) for i in range(dimension))


def _exercice_32() -> sp.Matrix:
    noyau = base_du_noyau(EQUATIONS_32)
    antecedents = (*noyau, *IMAGES_32)
    images = (*[(0, 0, 0, 0)] * len(noyau), *IMAGES_32.values())
    return ApplicationLineaire.depuis_les_images(antecedents, images).matrice


def toutes() -> dict[str, sp.Matrix]:
    """Chaque matrice que ces exercices demandent, par sa cle."""
    e3 = canonique(3)
    return {
        "27": F_27.matrice,
        "29Pve": matrice_de_passage(V_29, e3),
        "29Pev": matrice_de_passage(e3, V_29),
        "29fvv": PROJECTION_F,
        "29gvv": PROJECTION_G,
        "29fee": matrice_canonique(PROJECTION_F, V_29, V_29),
        "29gee": matrice_canonique(PROJECTION_G, V_29, V_29),
        "30a": matrice_associee(A_30, V_30, V_PRIME_30),
        "32": _exercice_32(),
    }


def noyau_et_rang_du_27() -> tuple[Base, int]:
    """Une base de `Ker f`, et le rang de `f`."""
    return F_27.noyau(), F_27.rang()


def reduction_du_30() -> tuple[sp.Matrix, sp.Matrix]:
    """`Q` et `P` inversibles telles que `Q A P = (I | 0)`.

    `Q` inverse le bloc des deux premieres colonnes ; `P` retranche de la
    troisieme ce qu'il en reste.
    """
    q = A_30[:, :2].inv()
    reste = q * A_30[:, 2]
    p = sp.eye(3)
    p[:2, 2] = -reste
    return q, p


def exercice_31() -> tuple[sp.Expr, ...]:
    """`f(x, y, z)`, sa matrice etant donnee de la base `w` vers la base `v`."""
    canonique_de_f = matrice_canonique(M_31, W_31, V_31)
    return tuple(sp.expand(canonique_de_f * M(sp.symbols("x y z"))))
