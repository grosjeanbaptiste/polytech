"""Le determinant par operations elementaires, par Sarrus, selon une colonne.

Le syllabus d'algebre (V.1, V.2) calcule un determinant en faisant
apparaitre des zeros avant de developper. Trois proprietes disent ce que
chaque operation fait au determinant :

1. permuter deux rangees le change de signe ;
2. multiplier une rangee par `alpha` le multiplie par `alpha` ;
3. `l_i -> l_i + beta l_k`, avec `i != k`, ne le change pas.

Une matrice triangulaire a pour determinant le produit de ses elements
diagonaux : echelonner jusqu'au bout suffit donc. Et pour une matrice
`3 x 3`, « et uniquement dans ce cas-la », la regle de Sarrus.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.algebre.echelonnement import echelonner


@dataclass(frozen=True, slots=True)
class Transformee:
    """La matrice obtenue, et `facteur` : `det(matrice) = facteur . det(origine)`."""

    matrice: sp.Matrix
    facteur: sp.Expr


def permuter(matrice: sp.Matrix, premiere: int, seconde: int) -> Transformee:
    """Propriete 1 : le determinant change de signe."""
    resultat = sp.Matrix(matrice)
    resultat.row_swap(premiere, seconde)
    return Transformee(resultat, sp.Integer(-1))


def multiplier(matrice: sp.Matrix, ligne: int, alpha: sp.Expr) -> Transformee:
    """Propriete 2 : le determinant est multiplie par `alpha`."""
    resultat = sp.Matrix(matrice)
    resultat[ligne, :] = alpha * matrice[ligne, :]
    return Transformee(resultat, sp.sympify(alpha))


def combiner(matrice: sp.Matrix, ligne: int, beta: sp.Expr, autre: int) -> Transformee:
    """Propriete 3 : `l_i -> l_i + beta l_k` laisse le determinant inchange."""
    if ligne == autre:
        msg = "une ligne ne se combine pas avec elle-meme : ce serait la multiplier"
        raise ValueError(msg)
    resultat = sp.Matrix(matrice)
    resultat[ligne, :] = matrice[ligne, :] + beta * matrice[autre, :]
    return Transformee(resultat, sp.Integer(1))


def _exiger_carree(matrice: sp.Matrix) -> None:
    if matrice.rows != matrice.cols:
        msg = "matrice non carree : elle n'a pas de determinant"
        raise ValueError(msg)


def du_triangulaire(matrice: sp.Matrix) -> sp.Expr:
    """Le produit des elements diagonaux d'une matrice triangulaire."""
    _exiger_carree(matrice)
    if not (matrice.is_upper or matrice.is_lower):
        msg = "matrice non triangulaire : le produit diagonal n'est pas son determinant"
        raise ValueError(msg)
    return sp.Mul(*(matrice[i, i] for i in range(matrice.rows)))


def par_echelonnement(matrice: sp.Matrix) -> sp.Expr:
    """Echelonner jusqu'au bout : un signe par permutation, puis le produit diagonal."""
    _exiger_carree(matrice)
    resultat = echelonner(matrice)
    permutations = sum(1 for operation in resultat.operations if "<->" in operation)
    return sp.simplify((-1) ** permutations * du_triangulaire(resultat.matrice))


def par_sarrus(matrice: sp.Matrix) -> sp.Expr:
    """Les trois diagonales descendantes moins les trois montantes."""
    if matrice.shape != (3, 3):
        msg = "la regle de Sarrus vaut uniquement pour une matrice 3 x 3"
        raise ValueError(msg)
    descendantes = sum(
        sp.Mul(*(matrice[i, (i + k) % 3] for i in range(3))) for k in range(3)
    )
    montantes = sum(
        sp.Mul(*(matrice[2 - i, (i + k) % 3] for i in range(3))) for k in range(3)
    )
    return descendantes - montantes


def selon_la_colonne(matrice: sp.Matrix, colonne: int) -> tuple[sp.Expr, ...]:
    """Les termes `a_ij C_ij` du developpement selon une colonne, non sommes."""
    _exiger_carree(matrice)
    if not 0 <= colonne < matrice.cols:
        msg = f"colonne hors de la matrice : {colonne} pour {matrice.cols} colonnes"
        raise ValueError(msg)
    return tuple(
        matrice[i, colonne] * matrice.cofactor(i, colonne) for i in range(matrice.rows)
    )
