"""Recueil d'algebre, chapitre VIII, exercices 17, 20, 21, 23 a 26.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, section VIII.7
(pages 113 a 115) : matrices de changement de base, et matrice associee
dans des bases qui ne sont pas canoniques.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application import recueil_algebre_ch08b_suite as suite
from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.familles_de_vecteurs import coordonnees
from polytech.maths.domain.algebre.representation import (
    matrice_associee,
    matrice_canonique,
    matrice_de_passage,
)

M = sp.Matrix
I: Final = sp.I  # noqa: E741  # l'unite imaginaire, que le syllabus note i
canonique = suite.canonique
F_17: Final = ApplicationLineaire.depuis_la_regle(lambda x, y, z: (x + y, x - z), 3)
V_17: Final = ((1, 1, 0), (0, 1, 1), (1, 1, 1))
W_17: Final = ((2, 1), (1, 1))
V_20: Final = ((1, 1, 0), (-1, 1, 1), (0, 1, 2))
V_PRIME_20: Final = ((2, 1, 1), (0, 0, 1), (-1, 1, 1))
X_20: Final = (1, 0, 0)
INTEGRATION_21: Final = M(5, 4, lambda i, j: sp.Rational(1, i) if i == j + 1 else 0)
V_21: Final = ((1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (0, 0, 1, 1))
V_PRIME_21: Final = (
    (1, 0, 0, 0, 0),
    (1, 1, 0, 0, 0),
    (1, 1, 1, 0, 0),
    (0, 0, 1, 1, 0),
    (0, 0, 0, 1, 1),
)
V_23: Final = ((1, 0, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0))
W_23: Final = ((I, 0, 0, 0), (0, 1, 0, 0), (0, 0, I, 0), (0, 0, 0, 1))
P_24: Final = M([[2, -1, 3], [1, 1, 4], [0, 1, 2]])
DERIVATION_25: Final = M([[0, 1, 0], [0, 0, 2]])
V_PRIME_25: Final = ((5, -2, 1), (0, -3, 2), (3, 1, 0))
W_PRIME_25: Final = ((-2, 2), (-3, 4))
V_26: Final = ((0, 2, 1), (1, 0, 3), (0, -1, 1))
DIAGONALE_26: Final = sp.diag(1, 1, -3)


def toutes() -> dict[str, sp.Matrix]:
    """Chaque matrice que les exercices 17 a 32 demandent, par sa cle."""
    matrices = {
        "17": matrice_associee(F_17.matrice, V_17, W_17),
        "20Pvv": matrice_de_passage(V_20, V_PRIME_20),
        "20Pvprimev": matrice_de_passage(V_PRIME_20, V_20),
        "21a": INTEGRATION_21,
        "21Pve": matrice_de_passage(V_21, canonique(4)),
        "21Pvprimee": matrice_de_passage(V_PRIME_21, canonique(5)),
        "21c": matrice_associee(INTEGRATION_21, V_21, V_PRIME_21),
        "23": matrice_de_passage(W_23, V_23),
        "24inverse": P_24.inv(),
        "25a": DERIVATION_25,
        "25Pv": matrice_de_passage(V_PRIME_25, canonique(3)),
        "25Pw": matrice_de_passage(W_PRIME_25, canonique(2)),
        "25c": matrice_associee(DERIVATION_25, V_PRIME_25, W_PRIME_25),
        "26vv": DIAGONALE_26,
        "26ee": matrice_canonique(DIAGONALE_26, V_26, V_26),
    }
    return matrices | suite.toutes()


def coordonnees_du_20() -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """Les coordonnees de `X` dans `v`, puis dans `v'` par la matrice de passage."""
    dans_v = coordonnees(X_20, V_20)
    dans_v_prime = matrice_de_passage(V_20, V_PRIME_20) * M(list(dans_v))
    return dans_v, tuple(dans_v_prime)
