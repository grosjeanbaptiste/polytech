"""L'exponentielle d'une matrice carree, `e^A = somme A^n / n!`.

La serie converge pour toute matrice, puisque `||A^n / n!|| <= ||A||^n / n!`.
Elle verifie `e^0 = I`, `e^I = e I`, `(e^A)^-1 = e^(-A)`, et
`e^A e^B = e^(A+B)` **quand `A` et `B` commutent**.

La serie n'est pas un procede de calcul. Si `A = P D P^-1` est
diagonalisable, `e^A = P e^D P^-1`, ou `e^D` porte les exponentielles des
valeurs propres. Sinon `A = P J P^-1` avec `J` sous forme de Jordan, et
l'on travaille bloc par bloc : pour un bloc de taille `n` et de valeur
propre `lambda`, `e^J` vaut `e^lambda` fois la matrice triangulaire des
`1/k!`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.jordan import forme_de_jordan
from polytech.maths.domain.algebre.spectre import Spectre


def par_la_serie(matrice: sp.Matrix, ordre: int) -> sp.Matrix:
    """`somme A^n / n!` jusqu'a `n = ordre` compris."""
    somme, terme = sp.eye(matrice.rows), sp.eye(matrice.rows)
    for n in range(1, ordre + 1):
        terme = terme * matrice / n
        somme += terme
    return somme.applyfunc(sp.expand)


def par_diagonalisation(matrice: sp.Matrix) -> sp.Matrix:
    """`P e^D P^-1` ; leve si la matrice n'est pas diagonalisable."""
    passage, diagonale = Spectre(matrice).diagonalisation()
    exponentielles = sp.diag(*(sp.exp(valeur) for valeur in diagonale.diagonal()))
    return sp.simplify(passage * exponentielles * passage.inv())


def exponentielle_d_un_bloc(valeur: object, taille: int) -> sp.Matrix:
    """`e^lambda` fois la matrice triangulaire superieure des `1/(k - j)!`."""
    return sp.exp(sp.sympify(valeur)) * sp.Matrix(
        taille,
        taille,
        lambda j, k: 1 / sp.factorial(k - j) if k >= j else 0,
    )


def _tailles_des_blocs(blocs: sp.Matrix) -> list[int]:
    """Les tailles des blocs de Jordan, lues sur la sur-diagonale."""
    tailles = [1]
    for i in range(blocs.rows - 1):
        if blocs[i, i + 1] == 0:
            tailles.append(1)
        else:
            tailles[-1] += 1
    return tailles


def par_jordan(matrice: sp.Matrix) -> sp.Matrix:
    """`P e^J P^-1`, l'exponentielle de `J` etant calculee bloc par bloc."""
    passage, blocs = forme_de_jordan(matrice)
    exponentielles, debut = [], 0
    for taille in _tailles_des_blocs(blocs):
        exponentielles.append(exponentielle_d_un_bloc(blocs[debut, debut], taille))
        debut += taille
    return sp.simplify(passage * sp.diag(*exponentielles) * passage.inv())
