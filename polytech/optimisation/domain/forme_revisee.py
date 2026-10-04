"""Extensions, partie I : les objets de la forme revisee de l'algorithme simplexe.

Le tableau simplexe recalcule tout a chaque iteration, y compris des
coefficients inutiles, et y perd les zeros de `T`. La forme revisee ne
garde que l'inverse de la base et retourne aux **donnees** pour ce dont
elle a besoin. Le probleme `z - c x = 0`, `T x = d` s'ecrit par le

    tableau des donnees   (T^, d^) = [[0, -c, 1], [d, T, 0]],

et une base `B` s'y etend de la ligne de `z` :

    B^ = [[1, -c_B], [0, B]],     B^^-1 = [[1, c_B B^-1], [0, B^-1]].

Le produit `B^^-1 (T^, d^)` est le tableau simplexe complet. La **forme
produit** evite d'inverser : apres un pivot, `B_t^-1 = E_t B_{t-1}^-1`,
ou `E_t` est l'identite dont une colonne est remplacee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.extensions import facteur_elementaire

if TYPE_CHECKING:
    from polytech.optimisation.domain.formes import FormeStandard


def tableau_des_donnees(forme: FormeStandard) -> sp.Matrix:
    """`[[0, -c, 1], [d, T, 0]]`."""
    m = forme.matrice.rows
    haut = sp.Matrix([[0]]).row_join(-forme.couts).row_join(sp.Matrix([[1]]))
    bas = forme.seconds_membres.row_join(forme.matrice).row_join(sp.zeros(m, 1))
    return haut.col_join(bas)


def inverse_de_base_etendue(forme: FormeStandard, base: tuple[int, ...]) -> sp.Matrix:
    """`[[1, c_B B^-1], [0, B^-1]]`, les indices de base comptes de 1."""
    colonnes = [i - 1 for i in base]
    matrice = forme.matrice[:, colonnes]
    if matrice.det() == 0:
        msg = f"{base} n'est pas une base : sous-matrice non reguliere"
        raise ValueError(msg)
    inverse = matrice.inv()
    m = len(base)
    haut = sp.Matrix([[1]]).row_join(forme.couts[:, colonnes] * inverse)
    return haut.col_join(sp.zeros(m, 1).row_join(inverse))


def facteur_etendu(colonne: sp.Matrix, ligne: int) -> sp.Matrix:
    """`E^_t` : la colonne entrante etendue de son `a_0l`, le pivot en `ligne`.

    C'est le facteur elementaire ordinaire, applique a la colonne dont la
    premiere composante est le coefficient de la ligne de `z`.
    """
    return facteur_elementaire(colonne, ligne)
