"""Une application lineaire de `K^n` dans `K^m`, son noyau et son image.

Le syllabus d'algebre (VIII.1) : `f : R^n -> R^m` est lineaire si et
seulement s'il existe une matrice `A` telle que `f(X) = A X`. La `j`-ieme
colonne de `A` est `f(e_j)`.

* le **noyau** est l'ensemble des `X` tels que `f(X) = 0` : les solutions
  du systeme homogene ;
* l'**image** est engendree par les images d'une famille generatrice du
  depart, donc par les colonnes de `A` ; on en extrait une base ;
* `dim Ker f + dim Im f = dim V`.

Une application entre espaces de matrices ou de polynomes se lit dans
`K^n` par les bijections de `isomorphismes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.algebre.familles_de_vecteurs import (
    est_une_base,
    extraire_une_base,
    matrice_des_lignes,
)
from polytech.maths.domain.algebre.sous_espaces import base_du_noyau

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polytech.maths.domain.algebre.familles_de_vecteurs import Famille, Vecteur


@dataclass(frozen=True, slots=True)
class ApplicationLineaire:
    """`X -> A X`, dans les bases canoniques."""

    matrice: sp.Matrix

    @classmethod
    def depuis_la_regle(
        cls, regle: Callable[..., Sequence[object]], dimension: int
    ) -> ApplicationLineaire:
        """La matrice dont les colonnes sont les images de la base canonique."""
        variables = sp.symbols(f"x1:{dimension + 1}")
        image = sp.Matrix(list(regle(*variables)))
        matrice = image.jacobian(variables)
        if not sp.simplify(matrice * sp.Matrix(variables) - image).is_zero_matrix:
            msg = "regle non lineaire : aucune matrice ne la represente"
            raise ValueError(msg)
        return cls(matrice)

    @classmethod
    def depuis_les_images(
        cls, antecedents: Famille, images: Famille
    ) -> ApplicationLineaire:
        """L'application que determinent les images d'une base."""
        if not est_une_base(antecedents):
            msg = "les antecedents ne forment pas une base : rien n'est determine"
            raise ValueError(msg)
        depart = matrice_des_lignes(antecedents).T
        return cls(matrice_des_lignes(images).T * depart.inv())

    def image_de(self, vecteur: Vecteur) -> tuple[sp.Expr, ...]:
        """`f(X) = A X`."""
        return tuple(sp.expand(self.matrice * sp.Matrix(list(vecteur))))

    def noyau(self) -> tuple[tuple[sp.Expr, ...], ...]:
        """Une base de `{X | A X = 0}`."""
        return base_du_noyau(self.matrice)

    def image(self) -> tuple[Vecteur, ...]:
        """Une base extraite des colonnes de `A`."""
        colonnes = [tuple(self.matrice.col(j)) for j in range(self.matrice.cols)]
        return extraire_une_base(colonnes)

    def rang(self) -> int:
        """La dimension de l'image."""
        return len(self.image())

    def est_injective(self) -> bool:
        """Le noyau est reduit au vecteur nul."""
        return not self.noyau()

    def est_surjective(self) -> bool:
        """L'image est l'espace d'arrivee tout entier."""
        return bool(self.rang() == self.matrice.rows)

    def inverse(self) -> ApplicationLineaire:
        """`f^-1`, de matrice `A^-1`."""
        if not (self.est_injective() and self.est_surjective()):
            msg = "application non inversible : elle n'est pas bijective"
            raise ValueError(msg)
        return ApplicationLineaire(self.matrice.inv())
