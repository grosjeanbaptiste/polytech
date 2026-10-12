"""Recueil d'algebre, chapitre VII : les familles de vecteurs des 44 exercices.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, sections VII.6 a
VII.10 (pages 82 a 93). Une matrice 2 x 2 se lit ligne apres ligne dans
`K^4` ; un polynome `a + bX + cX^2 + ...` est la suite de ses coefficients,
du terme independant au plus haut degre. C'est la bijection que le syllabus
emploie a ses exemples 10 et 11.

Ce fichier porte des donnees : une ligne par famille. Il depasse cent
lignes pour cette raison, et pour celle-la seule.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

I: Final = sp.I  # noqa: E741  # l'unite imaginaire, que le syllabus note i
R = sp.Rational
Vecteur = tuple[sp.Expr | int, ...]
Famille = tuple[Vecteur, ...]

PARAMETRE: Final = sp.Symbol("a")

VECTEURS: Final[dict[str, Famille]] = {
    "3a": ((1, 1, 0), (1, 1, 1), (0, 1, -1)),
    "3b": ((0, 1, 1), (0, 2, 1), (1, 5, 3)),
    "4": ((1, -1, 2, 1), (2, 0, -1, 0), (1, 3, -8, -3)),
    "5": ((1, 1, 0), (1, 1, 1), (PARAMETRE, 1, -1)),
    "6": ((2, 1, 1), (-1, 4, 5)),
    "7": ((3, 1, 2), (1, 3, 2), (5, 7, 6)),
    "8": ((1, 2, -1, 3), (2, 4, 1, -2), (3, 6, 3, -7)),
    "9": ((1, 1, 2, 1), (1, -1, 0, 1), (0, 0, -1, 1), (1, 2, 2, 0)),
    "10F1": ((1, 1, 0), (0, 1, 1), (1, 3, 2)),
    "10F2": ((1, 1, 0), (0, 1, 1), (0, 0, 1)),
    "10F3": ((1, 2, 1), (1, 0, 1)),
    "10F4": ((1, 1, 1), (2, 1, 0), (2, 0, 1), (0, 0, 1)),
    "10F5": ((1, 1, 1), (0, 1, 1), (0, 2, 2), (2, 3, 3)),
    "11X": ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)),
    "11Y": ((2, 5, 0, 7), (1, 1, 0, 1), (1, 3, 0, 2)),
    "12": ((1, -1, I), (-1, I, 1), (I, 1, -1)),
    "13a": ((0, 1, -1), (1, 1, 0), (1, 0, 2)),
    "13b": ((1, 1, 1), (-1, 1, 0), (1, 0, -1)),
    "14": ((1, -1, 2, 1), (2, 0, -1, 0), (1, 3, -8, -3)),
    "15": ((1, 5, 2, -1), (0, -3, 0, 1)),
    "17S": ((1, -1, 2, 1), (2, 0, -1, 0), (0, -2, 5, 1)),
    "17T": ((1, -1, 2, 0), (4, -2, 3, 1), (3, -1, 1, -1)),
    "19W1": ((1, 1, 0, -2), (0, -2, 1, 1)),
    "19W2": ((1, -2, 1, 0), (2, -3, 0, 1)),
    "20W1": ((2, 1),),
    "20W2": ((0, 1),),
    "20W3": ((1, 1),),
    "21W1": ((1, 0, 0),),
    "21W2": ((1, 1, 0), (0, 1, 1)),
    "22": ((1, 1, 1, 1), (-1, 1, 1, 1), (2, -1, -2, -1)),
}
"""Exercices 3 a 22 : des vecteurs de `R^n` (de `C^3` au 12)."""

CIBLES: Final[dict[str, Vecteur]] = {
    "6": (R(1, 6), R(7, 3), R(17, 6)),
    "7": (0, 0, 1),
    "9": (1, 1, 1, 1),
    "13a": (1, 1, 1),
    "13b": (0, 0, 1),
}
"""Le vecteur a ecrire dans la famille, ou a chercher dans son sous-espace."""

MATRICES: Final[dict[str, Famille]] = {
    "23": ((1, -2, 0, 3), (2, 3, 0, -1), (2, -1, 2, 1), (3, 9, -4, -2)),
    "25": ((1, 1, 1, 1), (1, 0, 0, 2), (2, -1, -1, 5)),
    "26": ((1, I, 3, -I), (I, 0, 2, I), (-1 + I, -1, 5 * I, 0)),
    "27": ((1, -2, 0, 3), (2, 3, 0, -1), (2, -1, 2, 1)),
    "29W1": ((0, 1, -1, 0),),
    "29W2": ((1, 2, 3, 4), (1, 1, 4, 4), (3, 5, 10, 12)),
}
"""Exercices 23 a 29 : des matrices 2 x 2, lues ligne apres ligne."""

POLYNOMES: Final[dict[str, Famille]] = {
    "31": ((1, 1, -1, 0, 1), (0, 0, 0, 1, -1), (1, 0, 0, 1, 0), (2, 1, -1, 3, -1)),
    "32F1": ((1, 1, -2), (1, -2, 1), (-2, 1, 1)),
    "32F2": ((3, 2, 1), (5, 3, 4), (1, 1, 1), (2, 1, 3), (1, 3, 2)),
    "32F3": ((0, 1, 0), (0, 0, 0), (1, 0, 1)),
    "32F4": ((1, 1, 0), (-1, 0, 1)),
    "33": ((1, -1, 1), (-1, 2, 3)),
    "34": ((5, -2, 1), (0, -3, 2), (3, 1, 0)),
    "35": ((1, 1, 1, 1), (1, 0, 0, 2), (2, -1, -1, 5)),
    "37": ((1, -1, 1, 1), (-1, 1, 2, 2), (2, -2, 1, 1)),
    "44": ((1, 1, 0, 1), (1, -1, 1, -2), (-1, 1, -1, 1)),
}
"""Exercices 31 a 37, et le 44 dans la base `(v1, v2, v3, v4)`."""
