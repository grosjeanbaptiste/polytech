"""Les enonces du chapitre X du recueil d'algebre : quarante exercices.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 157 et 174 a
183. Ce fichier ne porte que des donnees -- une matrice par exercice, et
les valeurs du parametre que chaque discussion distingue : il depasse cent
lignes pour cette raison, et chaque entree a la meme forme.

Les parametres sont reels : le chapitre travaille dans R, et c'est ce qui
permet de dire d'une valeur propre `a` qu'elle est reelle.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

M = sp.Matrix
T: Final = sp.Symbol("t")
A: Final = sp.Symbol("a", real=True)
B: Final = sp.Symbol("b", real=True)
ALPHA: Final = sp.Symbol("alpha", real=True)
BETA: Final = sp.Symbol("beta", real=True)
A_POSITIF: Final = sp.Symbol("a", positive=True)
"""Le meme `a`, sous l'hypothese `a > 0` de l'exercice 38."""

MATRICES: Final[dict[int, sp.Matrix]] = {
    1: M([[-15, 1, -9], [0, 6, 0], [4, 1, -3]]),
    2: M([[5, 7, 14, -8], [0, 3, 2, 0], [0, 0, 0, 2], [0, 0, 0, 7]]),
    3: M([[1, 2, 3], [3, 0, 3], [1, 4, 1]]),
    5: M([[2, 2], [1, 3]]),
    6: M([[2, 0, 1, 1], [-2, 2, 1, -1], [0, 0, 3, 1], [0, -2, -1, 3]]),
    7: M([[-15, 1, -9], [0, 6, 0], [4, 1, -3]]),
    8: M([[1, -3, 3], [3, -5, 3], [6, -6, 4]]),
    9: M([[3, -1, 0, 0], [0, 3, 0, 0], [1, 0, 3, 1], [0, 1, 0, 3]]),
    10: M([[-6, 4, 2, -1], [-10, 5, 0, 2], [-1, 2, 3, -3], [-3, 2, 0, 0]]),
    11: M([[2, 2, 1], [1, 3, 1], [1, 2, 2]]),
    12: M([[4, 2, 2], [2, 4, 2], [2, 2, 4]]),
    13: M([[1 - A, A], [B, 1 - B]]),
    14: M([[3, 2, 2, -4], [2, 3, 2, -1], [1, 1, 2, -1], [2, 2, 2, -1]]),
    16: M([[1, -1, -1, -1], [-1, -1, -1, 1], [-1, -1, 1, -1], [-1, 1, -1, -1]]) / 2,
    19: M(
        [
            [2, 1, 0, 1, -4],
            [0, 0, 0, 2, -2],
            [0, 2, 2, 0, 0],
            [0, 2, 0, 0, 2],
            [0, 0, 0, 0, -2],
        ]
    ),
    20: M([[ALPHA, 5, 7], [0, 3, 4], [-1, BETA, -4]]),
    21: M([[4, 5, -2], [-2, -2, 1], [-1, -1, 1]]),
    22: M([[0, 0, 2, 0], [1, 0, 1, 0], [0, 1, -2, 0], [0, 0, 0, 1]]),
    23: M([[3 - A, 2 - A, A - 2], [6 - 3 * A, A + 3, -3], [8 - 4 * A, 4, A - 4]]),
    24: M([[4 - A, 2 - A, A - 2], [6, 4, -3], [10 - 2 * A, 6 - 2 * A, 2 * A - 5]]),
    25: M([[BETA, 5, 7], [0, 3, ALPHA], [-1, -1, -4]]),
    26: M([[B, A, A], [-1, 1, -1], [1, 0, 2]]),
    27: M([[ALPHA + 1, -1, 1], [5, ALPHA - 4, 3], [7, -5, ALPHA + 3]]),
    28: M(
        [
            [3 - ALPHA, 1 - ALPHA, ALPHA - 1],
            [6 - 8 * ALPHA, 4 - 4 * ALPHA, 5 * ALPHA - 3],
            [7 - 9 * ALPHA, 3 - 5 * ALPHA, 6 * ALPHA - 2],
        ]
    ),
    29: M([[A, B, A], [-B, A - B, -1], [B, 2 * B, A + B]]),
    30: M([[0, 2, -2, 1], [4, 2, -4, 2], [3, 1, -3, 1], [-3, 2, -2, 0]]),
    31: M(
        [
            [B - 1, A + 1, A],
            [B - 2, A + 3, A + 1],
            [2 - B, B - A - 4, B - A - 2],
        ]
    ),
    32: M([[-2, 2, -2, 1], [-9, 8, -6, 3], [-8, 7, -5, 2], [-2, 4, -4, 1]]),
    33: M([[A, 3, B], [-1, 1, B - 4], [A - 1, 0, B - 1]]),
    34: M([[0, 1, 0], [0, 0, 1], [-A, 1, A]]),
    35: M([[3 * A, A + 1, -1], [4 - 5 * A, -A, 2 - A], [5 - 3 * A, 2 - A, 0]]),
    36: M([[0, 1, 0], [-1, A, 0], [0, 0, 1]]),
    37: M([[A - 1, 2, 1], [2 * A - 3, 4, 1], [-2 * A + 3, A - 3, A - 1]]),
    38: M([[1, 0, 1], [1 + A, -A, 1 + A], [1 + A, -1 - A, A]]),
    39: M([[A, A + 3, 2], [-A, A - 3, A - 2], [3 * A, -A + 6, 4 - 2 * A]]),
    40: M([[A + B + 1, 4 * B + 1, 2 * B], [-1, A - B - 1, -B], [A, A + B, A]]),
}
"""Numero de l'exercice : sa matrice. Les exercices 4, 15, 17 et 18 n'en donnent pas."""

CAS: Final[dict[int, dict[str, dict[sp.Symbol, object]]]] = {
    23: {"AUn": {A: 1}, "ADeux": {A: 2}},
    24: {"AUn": {A: 1}, "ADeux": {A: 2}},
    26: {"BUn": {B: 1}, "BUnAZero": {B: 1, A: 0}, "BDeux": {B: 2}},
    28: {"AlphaUn": {ALPHA: 1}},
    34: {"AUn": {A: 1}, "AMoinsUn": {A: -1}},
    36: {"AMoinsDeux": {A: -2}, "ADeux": {A: 2}},
    38: {"AZero": {A: 0}, "AMoinsUn": {A: -1}, "APositif": {A: A_POSITIF}},
    39: {"AZero": {A: 0}, "AUn": {A: 1}},
}
"""Numero : les valeurs du parametre que la discussion distingue."""

VECTEUR_3: Final = M([1, 1, 1])
ORDRE_4: Final = 5
RANG_4: Final = 2
MULTIPLICITE_4: Final = 2
VECTEUR_14: Final = M([0, 3, 1, 2])
DETERMINANT_14: Final = 6
VALEUR_14: Final = 1
CARRE_15: Final = M([[1, 0, 0], [-3, 1, -3], [3, 0, 4]])
NORMALE_18: Final = M([2, 1, -1])
"""Le plan `2x + y - z = 0`."""
DIRECTION_18: Final = M([1, -1, 3])
"""La droite `x = -y = z/3`."""
CIBLE_33: Final = T**3 - (B + 2) * T**2 + 8 * T - 4
SOMME_DES_CARRES_35: Final = 12
