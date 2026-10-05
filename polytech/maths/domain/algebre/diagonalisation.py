"""Sous-espaces propres, diagonalisabilite, matrices symetriques, puissances.

Le syllabus d'algebre (X.1, X.2) :

* les vecteurs propres associes a `lambda` sont les solutions non nulles
  de `(A - lambda I) X = 0` ; avec le vecteur nul ils forment le
  sous-espace propre `S(lambda)`, de dimension `n - rg(A - lambda I)` ;
* `A` est **diagonalisable** si et seulement si toutes les racines du
  polynome caracteristique sont reelles et que, pour chacune, la
  multiplicite geometrique egale la multiplicite algebrique ;
* une matrice **symetrique** reelle l'est toujours, et par une matrice
  orthogonale : `P^-1 = P^t`. Il suffit d'orthonormaliser la base de
  vecteurs propres, des vecteurs propres de valeurs distinctes etant deja
  orthogonaux ;
* si `P^-1 A P = D`, alors `A^k = P D^k P^-1`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.matrices_particulieres import est_symetrique
from polytech.maths.domain.algebre.orthogonalisation import gram_schmidt, normaliser
from polytech.maths.domain.algebre.produits_scalaires import usuel
from polytech.maths.domain.algebre.sous_espaces import base_du_noyau
from polytech.maths.domain.algebre.spectre import Spectre


def sous_espace_propre(
    matrice: sp.Matrix, valeur: object
) -> tuple[tuple[sp.Expr, ...], ...]:
    """Une base de `S(lambda)`, le noyau de `A - lambda I`."""
    base = base_du_noyau(matrice - sp.sympify(valeur) * sp.eye(matrice.rows))
    if not base:
        msg = f"{valeur} n'est pas une valeur propre : A - lambda I est inversible"
        raise ValueError(msg)
    return base


def est_diagonalisable(matrice: sp.Matrix) -> bool:
    """Racines toutes reelles, et `m_g = m_a` pour chacune."""
    return all(
        valeur.is_real and len(vecteurs) == algebrique
        for valeur, algebrique, vecteurs in matrice.eigenvects()
    )


def diagonaliser_orthogonalement(matrice: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    """`(P, D)` avec `P` orthogonale et `P^t A P = D` diagonale."""
    if not est_symetrique(matrice):
        msg = "matrice non symetrique : aucune matrice orthogonale ne la diagonalise"
        raise ValueError(msg)
    colonnes: list[sp.Matrix] = []
    valeurs: list[sp.Expr] = []
    for valeur in Spectre(matrice).valeurs_propres():
        base = [sp.Matrix(list(v)) for v in sous_espace_propre(matrice, valeur)]
        colonnes += normaliser(gram_schmidt(base, usuel), usuel)
        valeurs += [valeur] * len(base)
    return sp.Matrix.hstack(*colonnes), sp.diag(*valeurs)


def puissance(matrice: sp.Matrix, exposant: object) -> sp.Matrix:
    """`A^k = P D^k P^-1`, pour une matrice diagonalisable."""
    passage, diagonale = Spectre(matrice).diagonalisation()
    elevee = sp.diag(*(diagonale[i, i] ** exposant for i in range(diagonale.rows)))
    return sp.simplify(passage * elevee * passage.inv())
