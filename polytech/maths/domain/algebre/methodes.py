"""Les quatre methodes de resolution d'un systeme lineaire que le cours rappelle.

Sur le systeme `2x + y = 1`, `3x + 7y = -2`, les planches 4 a 7 deroulent :

* la **substitution** : isoler une inconnue d'une equation, la reporter
  dans l'autre ;
* le **pivot de Gauss** : garder `L_1`, remplacer `L_2` par une
  combinaison qui annule son premier coefficient -- sans fraction,
  `L_2 <- a_11 L_2 - a_21 L_1` -- jusqu'au systeme triangulaire ;
* les **matrices** : `X = A^-1 B`, avec pour l'ordre deux
  `A^-1 = 1/(ad - bc) [[d, -b], [-c, a]]` ;
* les formules de **Cramer** : chaque inconnue est un quotient de
  determinants, celui du numerateur ayant sa colonne remplacee par le
  second membre.

Les trois dernieres demandent un determinant non nul : le systeme a alors
une solution, et une seule.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.algebre.systemes import SystemeLineaire

ORDRE_DEUX = (2, 2)


@dataclass(frozen=True, slots=True)
class Echelonnement:
    """Les operations sur les lignes, le systeme triangulaire, sa solution."""

    operations: tuple[str, ...]
    triangulaire: SystemeLineaire
    solution: sp.Matrix


def _exiger_un_systeme_regulier(systeme: SystemeLineaire) -> sp.Expr:
    if systeme.matrice.rows != systeme.matrice.cols:
        forme = f"{systeme.matrice.rows} x {systeme.matrice.cols}"
        msg = f"systeme carre attendu, {forme} recu"
        raise ValueError(msg)
    determinant = systeme.matrice.det()
    if determinant == 0:
        msg = "determinant nul : le systeme n'a pas une solution unique"
        raise ValueError(msg)
    return determinant


def inverse_d_ordre_deux(matrice: sp.Matrix) -> sp.Matrix:
    """`1/(ad - bc) [[d, -b], [-c, a]]`."""
    if matrice.shape != ORDRE_DEUX:
        msg = f"matrice 2 x 2 attendue, {matrice.rows} x {matrice.cols} recue"
        raise ValueError(msg)
    a, b, c, d = matrice
    determinant = a * d - b * c
    if determinant == 0:
        msg = "determinant nul : la matrice n'a pas d'inverse"
        raise ValueError(msg)
    return sp.Matrix([[d, -b], [-c, a]]) / determinant


def par_l_inverse(systeme: SystemeLineaire) -> sp.Matrix:
    """`X = A^-1 B`."""
    _exiger_un_systeme_regulier(systeme)
    return systeme.matrice.inv() * systeme.second_membre


def par_cramer(systeme: SystemeLineaire) -> sp.Matrix:
    """`x_i = det(A_i) / det(A)`, `A_i` ayant le second membre pour colonne `i`."""
    determinant = _exiger_un_systeme_regulier(systeme)
    inconnues = []
    for colonne in range(systeme.matrice.cols):
        remplacee = systeme.matrice.copy()
        remplacee[:, colonne] = systeme.second_membre
        inconnues.append(remplacee.det() / determinant)
    return sp.Matrix(inconnues)


def par_le_pivot(systeme: SystemeLineaire) -> Echelonnement:
    """L'echelonnement sans fraction, puis la remontee."""
    _exiger_un_pivot_par_colonne(systeme)
    etendue = systeme.matrice.row_join(systeme.second_membre)
    operations: list[str] = []
    n = systeme.matrice.rows
    for k in range(n):
        if etendue[k, k] == 0:
            autre = next(i for i in range(k + 1, n) if etendue[i, k] != 0)
            etendue.row_swap(k, autre)
            operations.append(f"L_{k + 1} <-> L_{autre + 1}")
        for i in range(k + 1, n):
            if etendue[i, k] == 0:
                continue
            pivot, facteur = etendue[k, k], etendue[i, k]
            etendue[i, :] = pivot * etendue[i, :] - facteur * etendue[k, :]
            operations.append(f"L_{i + 1} <- {pivot} L_{i + 1} - {facteur} L_{k + 1}")
    triangulaire = SystemeLineaire(etendue[:, :-1], etendue[:, -1])
    solution = triangulaire.matrice.upper_triangular_solve(triangulaire.second_membre)
    return Echelonnement(tuple(operations), triangulaire, solution)


def _exiger_un_pivot_par_colonne(systeme: SystemeLineaire) -> None:
    carre = systeme.matrice.rows == systeme.matrice.cols
    if not carre or systeme.matrice.det() == 0:
        msg = "pas de solution unique : le pivot ne rend pas un systeme triangulaire"
        raise ValueError(msg)
