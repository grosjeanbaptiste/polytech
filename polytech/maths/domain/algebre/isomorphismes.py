"""Lire un espace de matrices ou de polynomes comme un `K^n`.

Le syllabus d'algebre (VII.7, VII.8) pose deux bijections qui respectent
l'addition et la multiplication par un scalaire :

    R^(2x2) -> R^4 : [[a, b], [c, d]] -> (a, b, c, d),
    R[X]<=2 -> R^3 : a + b X + c X^2 -> (a, b, c).

« L'espace vectoriel `R^4` est une autre facon de voir l'espace vectoriel
`R^(2x2)` » : une famille de matrices ou de polynomes est libre,
generatrice ou base exactement quand ses images le sont, et tout le
chapitre se rejoue sur des colonnes.

Un espace de fonctions n'a pas de base finie, et rien de tel n'y vaut.
Une famille finie de fonctions y est pourtant libre des que la matrice de
ses valeurs en autant de points est inversible : une combinaison nulle
s'annulerait en ces points, et le systeme n'aurait que la solution nulle.
La reciproque est fausse -- des points mal choisis ne prouvent rien.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence


def vecteur_d_une_matrice(matrice: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Les elements de la matrice, ligne apres ligne."""
    return tuple(matrice)


def matrice_d_un_vecteur(
    vecteur: Sequence[object], lignes: int, colonnes: int
) -> sp.Matrix:
    """La bijection reciproque, pour un format donne."""
    if len(vecteur) != lignes * colonnes:
        msg = f"{len(vecteur)} composantes pour un format {lignes} x {colonnes}"
        raise ValueError(msg)
    return sp.Matrix(lignes, colonnes, list(vecteur))


def vecteur_d_un_polynome(
    polynome: sp.Expr, degre_maximal: int, variable: sp.Symbol = VARIABLE
) -> tuple[sp.Expr, ...]:
    """Les coefficients, du terme independant a celui de degre maximal."""
    forme = sp.Poly(polynome, variable)
    if forme.degree() > degre_maximal:
        msg = f"polynome de degre superieur a {degre_maximal}"
        raise ValueError(msg)
    return tuple(forme.coeff_monomial(variable**k) for k in range(degre_maximal + 1))


def polynome_d_un_vecteur(
    vecteur: Sequence[object], variable: sp.Symbol = VARIABLE
) -> sp.Expr:
    """La bijection reciproque : `(a, b, c) -> a + b X + c X^2`."""
    return sp.Add(*(sp.sympify(c) * variable**k for k, c in enumerate(vecteur)))


def libres_par_evaluation(
    fonctions: Sequence[sp.Expr], points: Sequence[object], variable: sp.Symbol
) -> bool:
    """Vrai si les valeurs aux points prouvent la famille libre ; faux ne dit rien."""
    if len(points) != len(fonctions):
        msg = "il faut autant de points que de fonctions"
        raise ValueError(msg)
    valeurs = sp.Matrix(
        [[fonction.subs(variable, point) for fonction in fonctions] for point in points]
    )
    return bool(sp.simplify(valeurs.det()) != 0)
