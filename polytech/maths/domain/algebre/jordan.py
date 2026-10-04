"""Les multiplicites d'une valeur propre et la forme normale de Jordan.

La multiplicite **algebrique** `m` d'une valeur propre est celle de la
racine dans le polynome caracteristique ; sa multiplicite **geometrique**
`g` est la dimension de son espace propre, `dim ker(A - lambda I)`. On a
toujours `g <= m`. Si `g = m` pour chaque valeur propre, la matrice est
diagonalisable.

Sinon il manque des vecteurs propres, que l'on remplace par des vecteurs
propres **generalises** : `(A - lambda I) v_n = v_(n-1)`. La matrice se
ramene alors a sa forme normale de Jordan, `A = P J P^-1`, ou `J` porte
les valeurs propres sur sa diagonale et des 1 juste au-dessus, par blocs.
"""

from __future__ import annotations

import sympy as sp


def multiplicites(matrice: sp.Matrix) -> dict[sp.Expr, tuple[int, int]]:
    """`{valeur propre : (algebrique, geometrique)}`."""
    return {
        valeur: (int(algebrique), len(vecteurs))
        for valeur, algebrique, vecteurs in matrice.eigenvects()
    }


def vecteur_generalise(
    matrice: sp.Matrix, valeur: object, precedent: sp.Matrix
) -> sp.Matrix:
    """Un `v` tel que `(A - lambda I) v = precedent`."""
    decalee = matrice - sp.sympify(valeur) * sp.eye(matrice.rows)
    solution, libres = decalee.gauss_jordan_solve(precedent)
    return solution.subs(dict.fromkeys(libres, 0))


def forme_de_jordan(matrice: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """Le couple `(P, J)` tel que `A = P J P^-1`."""
    passage, blocs = matrice.jordan_form()
    return sp.Matrix(passage), sp.Matrix(blocs)
