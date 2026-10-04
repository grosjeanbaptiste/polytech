"""Valeurs propres, vecteurs propres, diagonalisation.

Le cours enseigne le geste : ecrire `det(A - lambda I) = 0`, en tirer les
racines, puis resoudre `(A - lambda I) v = 0` pour chacune. Le polynome
caracteristique est donc expose, et non seulement ses racines -- c'est la
ligne que le corrige doit montrer.

Chaque couple rendu se verifie par `A v = lambda v`, sans table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

INCONNUE: Final = sp.Symbol("lambda")
"""L'inconnue du polynome caracteristique, notee comme le cours la note."""


@dataclass(frozen=True, slots=True)
class Spectre:
    """Le spectre d'une matrice carree, et les gestes qui l'etablissent."""

    matrice: sp.Matrix

    @property
    def inconnue(self) -> sp.Symbol:
        """Le symbole du polynome caracteristique."""
        return INCONNUE

    def polynome_caracteristique(self) -> sp.Expr:
        """`det(A - lambda I)`, developpe."""
        taille = self.matrice.rows
        return sp.expand((self.matrice - INCONNUE * sp.eye(taille)).det())

    def valeurs_propres(self) -> tuple[sp.Expr, ...]:
        """Les racines du polynome caracteristique, chacune une fois."""
        racines = sp.solve(sp.Eq(self.polynome_caracteristique(), 0), INCONNUE)
        return tuple(sorted(racines, key=sp.default_sort_key))

    def couples_propres(self) -> tuple[tuple[sp.Expr, sp.Matrix], ...]:
        """Les couples (valeur, vecteur), un vecteur par valeur distincte."""
        return tuple(
            (valeur, vecteurs[0])
            for valeur, _, vecteurs in self.matrice.eigenvects()
            if vecteurs
        )

    def diagonalisation(self) -> tuple[sp.Matrix, sp.Matrix]:
        """Le couple (P, D) tel que `P D P^-1 = A` ; leve si impossible."""
        if not self.matrice.is_diagonalizable():
            msg = (
                "matrice non diagonalisable : elle n'a pas autant de vecteurs "
                "propres independants que sa taille"
            )
            raise ValueError(msg)
        passage, diagonale = self.matrice.diagonalize()
        return sp.Matrix(passage), sp.Matrix(diagonale)


def diagonalisee_par(matrice: sp.Matrix, passage: sp.Matrix) -> sp.Matrix:
    """`D = P^-1 A P`, pour une matrice `P` dont les colonnes sont propres.

    Leve si les colonnes ne forment pas une base, ou si le produit n'est
    pas diagonal : `P` ne porte alors pas des vecteurs propres de `A`.
    """
    if passage.det() == 0:
        msg = "les colonnes de P ne forment pas une base : determinant nul"
        raise ValueError(msg)
    produit = passage.inv() * matrice * passage
    if not produit.is_diagonal():
        msg = "les colonnes de P ne sont pas des vecteurs propres : P^-1 A P plein"
        raise ValueError(msg)
    return sp.Matrix(produit)
