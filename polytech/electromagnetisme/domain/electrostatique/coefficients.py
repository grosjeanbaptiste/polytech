"""Les trois especes d'equations electrostatiques d'un systeme de conducteurs.

Le syllabus d'electricite generale (3.9 a 3.13) : pour `n` conducteurs de
charges `Q_k` et de potentiels `U_k0` par rapport a l'enveloppe (ou a
l'infini),

* premiere espece : `U = K Q` (3.11.4), `K` etant les coefficients de
  potentiel ;
* deuxieme espece : `Q = Gamma U` (3.12.4), avec `Gamma = K^-1` (3.12.8) --
  capacites sur la diagonale, coefficients d'influence mutuelle ailleurs ;
* troisieme espece : `Q_i = somme C_ij U_ij` (3.13.2), ou les capacites
  partielles valent `C_i0 = somme_k Gamma_ik` et `C_ij = -Gamma_ij` (3.13.3).

L'identite de Gauss `somme Q'_k U_k0 = somme Q_k U'_k0` (3.9.1) rend `K` et
`Gamma` symetriques. L'energie vaut `W_e = (1/2) somme Q_k U_k0` (3.10.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Couple = tuple[int, int]
"""Deux conducteurs, numerotes de 1 a n ; 0 designe l'enveloppe ou l'infini."""


def _colonne(valeurs: Sequence[Any] | sp.Matrix) -> sp.Matrix:
    return sp.Matrix(list(valeurs))


@dataclass(frozen=True)
class SystemeDeConducteurs:
    """Un systeme de conducteurs, par sa matrice `K` des coefficients de potentiel."""

    potentiel: sp.Matrix

    def __post_init__(self) -> None:
        """Refuse une matrice qui ne peut pas etre celle d'un systeme reel."""
        if self.potentiel.rows != self.potentiel.cols:
            msg = f"matrice carree attendue, et non {self.potentiel.shape}"
            raise ValueError(msg)
        if sp.simplify(self.potentiel - self.potentiel.T) != sp.zeros(
            *self.potentiel.shape
        ):
            msg = "coefficients de potentiel non symetriques : (3.9.1) est violee"
            raise ValueError(msg)

    def potentiels(self, charges: Sequence[Any] | sp.Matrix) -> sp.Matrix:
        """`U = K Q` : les potentiels que ces charges imposent."""
        return self.potentiel * _colonne(charges)

    def capacites(self) -> sp.Matrix:
        """`Gamma = K^-1` : capacites et coefficients d'influence mutuelle."""
        return sp.simplify(self.potentiel.inv())

    def charges(self, potentiels: Sequence[Any] | sp.Matrix) -> sp.Matrix:
        """`Q = Gamma U` : les charges que ces potentiels imposent."""
        return self.capacites() * _colonne(potentiels)

    def capacites_partielles(self) -> dict[Couple, sp.Expr]:
        """Les `C_i0` vers l'enveloppe, puis les `C_ij` entre conducteurs (`i < j`)."""
        gamma = self.capacites()
        ordre = gamma.rows
        vers_l_enveloppe = {
            (i + 1, 0): sp.simplify(sum(gamma.row(i))) for i in range(ordre)
        }
        mutuelles = {
            (i + 1, j + 1): -gamma[i, j]
            for i in range(ordre)
            for j in range(i + 1, ordre)
        }
        return vers_l_enveloppe | mutuelles

    def energie(self, charges: Sequence[Any] | sp.Matrix) -> sp.Expr:
        """`W_e = (1/2) somme Q_k U_k0`."""
        colonne = _colonne(charges)
        return colonne.dot(self.potentiels(colonne)) / 2


def residu_de_l_identite_de_gauss(
    charges: Sequence[Any] | sp.Matrix,
    potentiels: Sequence[Any] | sp.Matrix,
    autres_charges: Sequence[Any] | sp.Matrix,
    autres_potentiels: Sequence[Any] | sp.Matrix,
) -> sp.Expr:
    """`somme Q'_k U_k0 - somme Q_k U'_k0` : nul entre deux etats d'un meme systeme."""
    croise = _colonne(autres_charges).dot(_colonne(potentiels))
    return croise - _colonne(charges).dot(_colonne(autres_potentiels))


def energie_par_les_capacites_partielles(
    partielles: Mapping[Couple, sp.Expr], potentiels: Sequence[Any] | sp.Matrix
) -> sp.Expr:
    """`W_e = (1/2) (somme C_k0 U_k0^2 + somme C_ij U_ij^2)` (3.13.7)."""
    valeurs = [sp.Integer(0), *potentiels]
    termes = (c * (valeurs[i] - valeurs[j]) ** 2 for (i, j), c in partielles.items())
    return sp.Add(*termes) / 2
