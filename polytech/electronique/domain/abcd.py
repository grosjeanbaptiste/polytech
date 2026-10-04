"""Les quadripoles en parametres ABCD (matrice de chaine).

`[V1, I1] = [[A, B], [C, D]] [V2, I2]`, le courant de sortie compte
sortant. L'interet de cette ecriture est la **cascade** : deux
quadripoles l'un derriere l'autre ont pour matrice le produit des leurs,
dans l'ordre ou le signal les traverse.

Deux briques suffisent a tout construire : une impedance en serie
`[[1, Z], [0, 1]]`, une impedance en derivation `[[1, 0], [1/Z, 1]]`.
Chargee par `Z_L`, la sortie vaut `V2 = Z_L I2`, d'ou le gain en tension
`Z_L / (A Z_L + B)`.
"""

from __future__ import annotations

import sympy as sp


def serie(impedance: sp.Expr) -> sp.Matrix:
    """Une impedance traversee par le courant : `[[1, Z], [0, 1]]`."""
    return sp.Matrix([[1, impedance], [0, 1]])


def derivation(impedance: sp.Expr) -> sp.Matrix:
    """Une impedance entre les deux fils : `[[1, 0], [1/Z, 1]]`."""
    return sp.Matrix([[1, 0], [1 / impedance, 1]])


def cascade(*quadripoles: sp.Matrix) -> sp.Matrix:
    """Le produit, dans l'ordre de traversee ; leve sans quadripole."""
    if not quadripoles:
        msg = "aucun quadripole a mettre en cascade"
        raise ValueError(msg)
    produit = quadripoles[0]
    for suivant in quadripoles[1:]:
        produit = produit * suivant
    return sp.simplify(produit)


def cellule_en_t(entree: sp.Expr, sortie: sp.Expr, branche: sp.Expr) -> sp.Matrix:
    """`Z1` et `Z2` en serie, `Z3` en derivation entre les deux, par formules."""
    return sp.Matrix(
        [
            [1 + entree / branche, entree + sortie + entree * sortie / branche],
            [1 / branche, 1 + sortie / branche],
        ]
    )


def gain_en_tension(matrice: sp.Matrix, charge: sp.Expr) -> sp.Expr:
    """`Z_L / (A Z_L + B)` : la sortie chargee par `Z_L`."""
    return sp.simplify(charge / (matrice[0, 0] * charge + matrice[0, 1]))
