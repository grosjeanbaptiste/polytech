"""L'echelonnement d'une matrice, et l'inverse par la matrice-compagnon.

Une matrice est **echelonnee** si toute ligne nulle n'est suivie que de
lignes nulles et si la colonne du premier element non nul de chaque ligne
(son pivot) croit strictement. Elle est **echelonnee simple** si, de plus,
chaque pivot vaut 1 et n'a que des zeros au-dessus de lui.

La demarche du syllabus (II.7.2) : colonne par colonne, choisir un pivot
non nul -- en permutant deux lignes s'il n'est pas a sa place --, puis
faire apparaitre des zeros sous lui par `l_i -> l_i - (a_ik / pivot) l_k`.
Une colonne sans pivot possible est sautee. Pour l'echelonnement simple, on
remonte de droite a gauche en annulant au-dessus de chaque pivot, puis on
divise chaque ligne par son pivot.

Echelonner `[A | I]` jusqu'a `[I | B]` rend `B = A^-1` ; si la forme
echelonnee simple de `A` n'est pas l'identite, `A` n'est pas inversible.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class MatriceEchelonnee:
    """Les operations sur les lignes, dans l'ordre, et la matrice obtenue."""

    operations: tuple[str, ...]
    matrice: sp.Matrix


def colonnes_des_pivots(matrice: sp.Matrix) -> list[int | None]:
    """Pour chaque ligne, la colonne de son premier element non nul."""
    return [
        next((j for j in range(matrice.cols) if matrice[i, j] != 0), None)
        for i in range(matrice.rows)
    ]


def est_echelonnee(matrice: sp.Matrix) -> bool:
    """Les lignes nulles en bas, et des pivots de colonne strictement croissante."""
    precedent: int | None = -1
    for pivot in colonnes_des_pivots(matrice):
        if pivot is not None and (precedent is None or pivot <= precedent):
            return False
        precedent = pivot
    return True


def est_echelonnee_simple(matrice: sp.Matrix) -> bool:
    """Echelonnee, chaque pivot egal a 1 et seul non nul de sa colonne."""
    if not est_echelonnee(matrice):
        return False
    return all(
        matrice[i, pivot] == 1 and sum(1 for x in matrice[:, pivot] if x != 0) == 1
        for i, pivot in enumerate(colonnes_des_pivots(matrice))
        if pivot is not None
    )


def _combinaison(cible: int, coefficient: sp.Expr, source: int) -> str:
    signe = "-" if coefficient.could_extract_minus_sign() else "+"
    valeur = -coefficient if signe == "-" else coefficient
    facteur = "" if valeur == 1 else f"{valeur} "
    return f"l{cible + 1} -> l{cible + 1} {signe} {facteur}l{source + 1}"


def echelonner(matrice: sp.Matrix) -> MatriceEchelonnee:
    """La demarche II.7.2, operation par operation."""
    travail, operations, ligne = matrice.copy(), [], 0
    for colonne in range(travail.cols):
        candidates = [i for i in range(ligne, travail.rows) if travail[i, colonne] != 0]
        if not candidates:
            continue
        if candidates[0] != ligne:
            travail.row_swap(ligne, candidates[0])
            operations.append(f"l{ligne + 1} <-> l{candidates[0] + 1}")
        for i in candidates[1:]:
            coefficient = -travail[i, colonne] / travail[ligne, colonne]
            travail[i, :] = sp.simplify(travail[i, :] + coefficient * travail[ligne, :])
            operations.append(_combinaison(i, coefficient, ligne))
        ligne += 1
    return MatriceEchelonnee(tuple(operations), travail)


def echelonner_simplement(matrice: sp.Matrix) -> sp.Matrix:
    """Des zeros au-dessus de chaque pivot, de droite a gauche, puis des pivots 1."""
    travail = echelonner(matrice).matrice
    for i, pivot in reversed(list(enumerate(colonnes_des_pivots(travail)))):
        if pivot is None:
            continue
        travail[i, :] = sp.simplify(travail[i, :] / travail[i, pivot])
        for dessus in range(i):
            travail[dessus, :] = sp.simplify(
                travail[dessus, :] - travail[dessus, pivot] * travail[i, :]
            )
    return travail


def inverse_par_la_matrice_compagnon(matrice: sp.Matrix) -> sp.Matrix:
    """`[A | I]` echelonnee simplement en `[I | A^-1]`."""
    if matrice.rows != matrice.cols:
        msg = "matrice non carree : elle n'a pas d'inverse"
        raise ValueError(msg)
    ordre = matrice.rows
    simple = echelonner_simplement(matrice.row_join(sp.eye(ordre)))
    if simple[:, :ordre] != sp.eye(ordre):
        msg = "la forme echelonnee simple n'est pas l'identite : matrice non inversible"
        raise ValueError(msg)
    return simple[:, ordre:]
