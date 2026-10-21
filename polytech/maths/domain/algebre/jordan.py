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

Les tailles des blocs se lisent sur les espaces propres generalises
`E_k = ker(A - lambda I)^k`, emboites, dont la dimension croit jusqu'a la
multiplicite algebrique (syllabus de mathematiques appliquees, 7.4.4) :
`dim E_k - dim E_(k-1)` blocs ont une taille d'au moins `k`.
"""

from __future__ import annotations

from itertools import pairwise

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


def dimensions_des_espaces_generalises(
    matrice: sp.Matrix, valeur: object
) -> tuple[int, ...]:
    """`dim E_1, dim E_2, ...`, jusqu'a ce que `E_k` ne s'accroisse plus."""
    decalee = matrice - sp.sympify(valeur) * sp.eye(matrice.rows)
    dimensions = [matrice.rows - decalee.rank()]
    if dimensions[0] == 0:
        msg = f"{valeur} n'est pas valeur propre : ker(A - lambda I) = {{0}}"
        raise ValueError(msg)
    while True:
        suivante = matrice.rows - (decalee ** (len(dimensions) + 1)).rank()
        if suivante == dimensions[-1]:
            return tuple(dimensions)
        dimensions.append(suivante)


def tailles_des_blocs(matrice: sp.Matrix, valeur: object) -> tuple[int, ...]:
    """Les tailles des blocs de Jordan de la valeur propre, de la plus grande."""
    dimensions = (0, *dimensions_des_espaces_generalises(matrice, valeur))
    au_moins = [b - a for a, b in pairwise(dimensions)]
    tailles: list[int] = []
    for k, nombre in enumerate(au_moins, start=1):
        suivant = au_moins[k] if k < len(au_moins) else 0
        tailles.extend([k] * (nombre - suivant))
    return tuple(sorted(tailles, reverse=True))


def ordre_de_la_reduction(matrice: sp.Matrix) -> int:
    """La somme des multiplicites algebriques des valeurs propres reelles.

    Formulaire, 1.2 : `A` est triangularisable quand elle vaut `n` ; « sinon
    A est reductible jusqu'a l'ordre » qu'elle donne.
    """
    return sum(
        int(algebrique)
        for valeur, algebrique in matrice.eigenvals().items()
        if valeur.is_real
    )
