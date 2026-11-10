"""Limites latérales, continuité en un point et espèces de discontinuité.

Syllabus de calcul différentiel et intégral 1, sections 4.1.5 et 4.2 :
une fonction est continue en `a` quand sa limite y vaut sa valeur
(définition 4.24). Si elle ne l'est pas et que les limites à gauche et à
droite existent dans `R`, la discontinuité est de **première espèce** ;
elle est **éliminable** si ces deux limites sont égales, et il suffit alors
de changer la valeur en `a`. Si l'une des limites n'existe pas dans `R`
(infinie ou oscillante), elle est de **deuxième espèce** (définition 4.25).

La valeur en `a` se donne à part : l'expression décrit la fonction hors de
`a`, comme les exemples 4.26 qui la redéfinissent en ce point.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X


class Discontinuite(Enum):
    """Ce qu'est une fonction en un point, au sens de la définition 4.25."""

    CONTINUE = "continue"
    ELIMINABLE = "discontinuite eliminable"
    PREMIERE_ESPECE = "discontinuite de premiere espece"
    DEUXIEME_ESPECE = "discontinuite de deuxieme espece"


def limite_a_gauche(expression: sp.Expr, point: object) -> sp.Expr:
    """La limite quand `x` tend vers `point` par la gauche."""
    return sp.limit(expression, X, point, "-")


def limite_a_droite(expression: sp.Expr, point: object) -> sp.Expr:
    """La limite quand `x` tend vers `point` par la droite."""
    return sp.limit(expression, X, point, "+")


def _reelle(valeur: sp.Expr) -> bool:
    """Vrai si la limite existe dans `R` : ni infinie, ni oscillante, ni indéfinie."""
    return not isinstance(valeur, sp.AccumBounds) and bool(valeur.is_finite)


def limite(expression: sp.Expr, point: object) -> sp.Expr:
    """La limite en `point` ; leve si les limites laterales different (4.1.5)."""
    gauche = limite_a_gauche(expression, point)
    droite = limite_a_droite(expression, point)
    if gauche != droite:
        msg = (
            f"{expression} n'a pas de limite en {point} : "
            f"limites laterales {gauche}, {droite}"
        )
        raise ValueError(msg)
    return gauche


def classer(expression: sp.Expr, point: object, *, valeur: object) -> Discontinuite:
    """L'espèce de la discontinuité en `point`, ou `CONTINUE` s'il n'y en a pas."""
    gauche = limite_a_gauche(expression, point)
    droite = limite_a_droite(expression, point)
    if not (_reelle(gauche) and _reelle(droite)):
        return Discontinuite.DEUXIEME_ESPECE
    if gauche != droite:
        return Discontinuite.PREMIERE_ESPECE
    if gauche == sp.sympify(valeur):
        return Discontinuite.CONTINUE
    return Discontinuite.ELIMINABLE


def est_continue_en(expression: sp.Expr, point: object, *, valeur: object) -> bool:
    """Vrai si `f` est continue en `point` (définition 4.24)."""
    return classer(expression, point, valeur=valeur) is Discontinuite.CONTINUE
