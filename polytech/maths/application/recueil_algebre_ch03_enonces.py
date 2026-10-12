"""Recueil d'algebre, chapitre III : les vingt-quatre systemes de la section III.7.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 41 a 44. Chaque
systeme est sa matrice augmentee, le second membre en derniere colonne.

Ce fichier porte des donnees : une ligne par equation, une table par famille.
Il depasse cent lignes pour cette raison, et pour celle-la seule.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

M = sp.Matrix
SYMBOLES: Final = sp.symbols("k mu lambda t u a b m")
"""Les parametres des exercices 13 a 24, dans l'ordre ou ils paraissent."""
K, MU, LAM, T, U, A, B, MM = SYMBOLES
Substitution = dict[sp.Symbol, int]

NUMERIQUES: Final = {
    "1": M([[2, 1, 3, 1], [4, 1, 3, 1], [3, 1, 3, 1]]),
    "2": M([[1, 1, 0, 3], [2, 3, 1, 9], [-1, 3, 4, 2]]),
    "3": M([[2, 4, 3, 1], [-1, 2, 1, 1], [2, 3, 2, 1]]),
    "4": M([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]]),
    "5": M([[1, 2, 1, 2, -3], [3, 6, 4, -1, 2], [4, 8, 5, 1, -1], [-2, -4, -3, 3, -5]]),
    "6": M(
        [
            [1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, -2, -1, 2, 0],
            [-3, 0, -3, -1, -1, -2, 0],
            [-1, 0, -1, 0, 0, -1, 0],
        ]
    ),
    "7": M([[1, -1, 1, 2, 3], [2, -1, 3, 6, 9], [3, -2, 4, 8, 12], [1, 0, 2, 4, 6]]),
    "8": M([[1, 1, 1, 1, 0], [1, 3, 2, 4, 0], [2, 0, 1, -1, 0]]),
    "9": M([[1, 1, 2, -3, 1], [2, 3, 6, -5, 2], [3, -1, 2, -7, 5], [1, 2, 3, -1, -1]]),
    "10a": M([[1, 1, 2, 1], [3, 2, 3, 4], [5, 4, 7, 6]]),
    "10b": M([[1, 1, 2, 3], [3, 2, 3, -3], [5, 4, 7, 3]]),
    "11": M(
        [[2, 1, -1, 1, 1], [3, -2, 2, -3, 2], [5, 1, -1, 2, -1], [2, -1, 1, -3, 4]]
    ),
    "12": M([[1, -1, 4, 3], [3, -9, 12, 4]]),
}
"""Exercices 1 a 12 ; le 10 pose deux systemes de meme matrice."""

PARAMETRIQUES: Final = {
    13: M([[1, 2, K, 1], [2, K, 8, 3]]),
    14: M([[2, -1, 3, 1], [-4, 2, 1, 3], [-2, 1, 4, MU], [10, -5, -6, LAM]]),
    15: M([[1, T, 1, U], [T, 1, T, 1], [1, T, -T, U]]),
    16: M([[A, B, 2, 1], [A, 2 * B - 1, 3, 1], [A, B, B + 3, 2 * B - 1]]),
    17: M([[LAM - 2, 2, -1, A], [2, LAM, 2, 1], [2 * LAM, 2 * (LAM + 1), LAM + 1, A]]),
    18: M([[B, A, 1, 1], [1, A * B, 1, A], [1, A, B, 1]]),
    19: M([[2, 2 * A, 1 - A, 2 * B], [A, 1, A, 1], [1, A, 1, B]]),
    20: M(
        [
            [A - 2, A - 2, 0, 2 * A - 2],
            [2, A - 1, A - 3, A + 5],
            [3 - A, 2 * A - 3, 2 * A - 2, A - 2],
        ]
    ),
    21: M([[MM - 1, -3, 0, 2], [2 * MM, MM - 1, 3 * MM - 1, 2 * T], [1, 3, MM, T - 1]]),
    22: M([[1, 1, B + 1, 1], [1, B, 1, A], [B, 1, -1, A**2]]),
    23: M([[1, A, 1, 1, B], [A, 1, 1, 1, 0], [1, 1, A, 1, 1], [1, 1, 1, A, 2]]),
    24: M([[1, -3, -2, 4], [3, -4, -9, 5], [4, -7, A**2 - 20, A + 6]]),
}
"""Exercices 13 a 24, a discuter selon un ou deux parametres."""

FORMES: Final[dict[int, dict[str, Substitution]]] = {
    13: {"General": {}},
    14: {"General": {}},
    15: {"TMoinsUn": {T: -1}, "TUn": {T: 1}},
    16: {"BMoinsUn": {B: -1}, "BUn": {B: 1}},
    17: {"LambdaZero": {LAM: 0}, "LambdaUn": {LAM: 1}, "LambdaDeux": {LAM: 2}},
    18: {"BUn": {B: 1}, "BMoinsDeux": {B: -2}},
    19: {"AMoinsUn": {A: -1}, "AUn": {A: 1}},
    20: {"ADeux": {A: 2}, "ATrois": {A: 3}, "AQuatre": {A: 4}},
    21: {"MZero": {MM: 0}, "MUn": {MM: 1}, "MQuatre": {MM: 4}},
    22: {"BUn": {B: 1}, "BMoinsUn": {B: -1}},
    23: {"AUn": {A: 1}, "AMoinsTrois": {A: -3}},
    24: {"ATrois": {A: 3}, "AMoinsTrois": {A: -3}},
}
"""Les substitutions dont le corrige montre la forme echelonnee."""

SOLUTIONS: Final[dict[int, dict[str, tuple[Substitution, tuple[int, ...]]]]] = {
    13: {"General": ({}, ())},
    14: {"Compatible": ({MU: 4, LAM: -10}, ())},
    15: {"TMoinsUn": ({T: -1, U: -1}, ()), "TUn": ({T: 1, U: 1}, ())},
    16: {"BUn": ({B: 1}, (0,)), "AZeroBCinq": ({A: 0, B: 5}, ())},
    17: {"LambdaUn": ({LAM: 1, A: -8}, ())},
    18: {"AUnBUn": ({A: 1, B: 1}, ()), "MoinsDeux": ({A: -2, B: -2}, ())},
    19: {"AMoinsUn": ({A: -1, B: -1}, ()), "AUn": ({A: 1, B: 1}, ())},
    20: {"ATrois": ({A: 3}, ()), "AZero": ({A: 0}, ())},
    21: {
        "MZero": ({MM: 0, T: -1}, ()),
        "MQuatre": ({MM: 4, T: -1}, ()),
        "MDeux": ({MM: 2}, ()),
    },
    22: {
        "BUnAUn": ({B: 1, A: 1}, ()),
        "BUnADeux": ({B: 1, A: 2}, ()),
        "BMoinsUnAZero": ({B: -1, A: 0}, ()),
        "BMoinsUnAMoinsUn": ({B: -1, A: -1}, ()),
    },
    23: {"AMoinsTrois": ({A: -3, B: -3}, ()), "AZero": ({A: 0}, ())},
    24: {"ATrois": ({A: 3}, ())},
}
"""Les cas compatibles : la substitution, et les inconnues imposees libres."""

A_SOLUTION_UNIQUE: Final = (15, 16, 17, 18, 19, 22, 24)
"""Les exercices dont le corrige donne la solution unique en toute generalite."""
