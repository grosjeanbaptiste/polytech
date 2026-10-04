"""Le determinant par developpement en cofacteurs, et la matrice adjointe.

Sympy calcule un determinant d'un trait. Le cours, lui, enseigne le
developpement selon une ligne -- et c'est cette suite de mineurs qu'un
corrige pas a pas doit montrer. D'ou `termes_du_developpement`, qui rend
les termes un par un, avant toute somme.

L'identite `A adj(A) = det(A) I` est la verification que le fascicule
emploie lui-meme avant d'ecrire l'inverse. Elle est exposee pour la meme
raison : elle se montre, et elle se verifie.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class Determinant:
    """Le determinant d'une matrice carree, et les gestes qui l'etablissent."""

    matrice: sp.Matrix

    def __post_init__(self) -> None:
        """Refuse la matrice non carree : elle n'a pas de determinant."""
        if self.matrice.rows != self.matrice.cols:
            msg = (
                f"matrice non carree : {self.matrice.rows} lignes "
                f"pour {self.matrice.cols} colonnes"
            )
            raise ValueError(msg)

    def _exiger_la_ligne(self, ligne: int) -> None:
        if not 0 <= ligne < self.matrice.rows:
            msg = f"ligne hors de la matrice : {ligne} pour {self.matrice.rows} lignes"
            raise ValueError(msg)

    def termes_du_developpement(self, ligne: int) -> tuple[sp.Expr, ...]:
        """Les termes `(-1)^(i+j) a_ij M_ij`, un par colonne, non sommes."""
        self._exiger_la_ligne(ligne)
        return tuple(
            (-1) ** (ligne + colonne)
            * self.matrice[ligne, colonne]
            * self.matrice.minor_submatrix(ligne, colonne).det()
            for colonne in range(self.matrice.cols)
        )

    def par_cofacteurs(self, ligne: int) -> sp.Expr:
        """Le determinant, obtenu en sommant le developpement."""
        return sp.simplify(sum(self.termes_du_developpement(ligne)))

    def adjointe(self) -> sp.Matrix:
        """La transposee de la comatrice, celle que le fascicule note `A tilde`."""
        return self.matrice.adjugate()

    def comatrice(self) -> sp.Matrix:
        """La matrice des cofacteurs `(-1)^(i+j) M_ij`, notee `A~` aux exercices."""
        return self.matrice.cofactor_matrix()

    def inverse(self) -> sp.Matrix:
        """`A^-1 = tA~ / det A` ; leve si le determinant est nul."""
        determinant = sp.simplify(self.matrice.det())
        if determinant == 0:
            msg = "determinant nul : la matrice n'a pas d'inverse"
            raise ValueError(msg)
        return self.comatrice().T / determinant

    def condition_de_singularite(self) -> sp.Eq:
        """`det A = 0`, factorise : la condition pour des solutions non nulles."""
        return sp.Eq(sp.factor(self.matrice.det()), 0)

    def valeurs_singulieres(self, parametre: sp.Symbol) -> tuple[sp.Expr, ...]:
        """Les valeurs du parametre qui annulent le determinant."""
        racines = sp.solve(sp.Eq(self.matrice.det(), 0), parametre)
        return tuple(sorted(racines, key=sp.default_sort_key))
