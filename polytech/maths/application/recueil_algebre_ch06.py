"""Recueil d'algebre, chapitre VI : le rang de cinq matrices.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, section VI.4 (page
69). Deux matrices chiffrees, dont le rang se lit sur la forme echelonnee ;
trois matrices carrees a parametre, dont le rang ne chute qu'aux racines du
determinant.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.echelonnement import MatriceEchelonnee, echelonner
from polytech.maths.domain.algebre.rang_exact import (
    rang_par_echelonnement,
    rangs_selon_le_parametre,
)

M = sp.Matrix
A: Final = sp.Symbol("a")
T: Final = sp.Symbol("t")
NUMERIQUES: Final = {
    "a": M([[-1, 2, 0, 4, 5], [3, -7, 2, 0, 1], [2, -5, 2, 4, 6], [4, -9, 2, -4, -4]]),
    "b": M([[1, -3, 2, 2], [0, 3, 6, 0], [2, -3, -2, 4], [3, -6, 0, 6]]),
}
PARAMETRIQUES: Final = {
    "c": (M([[1, 2 * A, 1], [A, 2, 1], [1, 2, A]]), A),
    "d": (M([[1, T, 1], [T, 1, T], [1, T, -T]]), T),
    "e": (M([[T, 3, -1], [3, 6, -2], [-1, -3, T]]), T),
}
"""Lettre de l'enonce : la matrice et son parametre."""
SINGULIERS: Final = {
    "cUn": ("c", sp.Integer(1)),
    "cMoinsDeux": ("c", sp.Integer(-2)),
    "dMoinsUn": ("d", sp.Integer(-1)),
    "dUn": ("d", sp.Integer(1)),
    "eUn": ("e", sp.Integer(1)),
    "eTroisDemis": ("e", sp.Rational(3, 2)),
}
"""Cle du cas : la matrice concernee et la valeur du parametre."""


def echelonnees() -> dict[str, MatriceEchelonnee]:
    """Les deux matrices chiffrees, echelonnees, avec leur journal."""
    return {lettre: echelonner(m) for lettre, m in NUMERIQUES.items()}


def rangs_numeriques() -> dict[str, int]:
    """Le nombre de lignes non nulles de chaque forme echelonnee."""
    return {lettre: rang_par_echelonnement(m) for lettre, m in NUMERIQUES.items()}


def determinants() -> dict[str, sp.Expr]:
    """Le determinant factorise des trois matrices a parametre."""
    return {k: sp.factor(m.det()) for k, (m, _) in PARAMETRIQUES.items()}


def rangs_a_parametre() -> dict[str, tuple[int, dict[sp.Expr, int]]]:
    """Le rang generique, et le rang en chaque valeur ou il chute."""
    return {k: rangs_selon_le_parametre(m, p) for k, (m, p) in PARAMETRIQUES.items()}


def formes_singulieres() -> dict[str, sp.Matrix]:
    """La forme echelonnee de chaque matrice en chacune de ses valeurs singulieres."""
    formes = {}
    for cle, (lettre, valeur) in SINGULIERS.items():
        matrice, parametre = PARAMETRIQUES[lettre]
        formes[cle] = echelonner(matrice.subs(parametre, valeur)).matrice
    return formes
