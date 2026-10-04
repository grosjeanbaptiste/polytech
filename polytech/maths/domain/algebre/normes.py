"""La norme d'une matrice carree, et la distance qu'elle definit.

    ||A|| = racine(somme des a_jk^2),

l'analogue de la norme euclidienne d'un vecteur. Elle est positive, nulle
pour la seule matrice nulle, homogene, et verifie `||A + B|| <= ||A|| +
||B||` ainsi que `||A B|| <= ||A|| ||B||` -- d'ou `||A^n|| <= ||A||^n`.

Avec `d(A, B) = ||A - B||`, les matrices forment un espace metrique : une
suite de matrices converge quand ses `N^2` suites de coefficients
convergent.
"""

from __future__ import annotations

import sympy as sp


def norme(matrice: sp.Matrix) -> sp.Expr:
    """`racine(somme a_jk^2)`."""
    return sp.simplify(sp.sqrt(sum((e**2 for e in matrice), sp.Integer(0))))


def distance(premiere: sp.Matrix, seconde: sp.Matrix) -> sp.Expr:
    """`d(A, B) = ||A - B||`."""
    return norme(premiere - seconde)
