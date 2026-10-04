"""L'espace vectoriel `R^n`, tel que la planche 28 le rappelle.

« Un espace vectoriel est un ensemble muni d'une structure permettant
d'effectuer des combinaisons lineaires » : `k x` et `x + y` restent dans
l'espace. Deux vecteurs non nuls sont **colineaires** quand l'un est un
multiple de l'autre -- c'est ce que dit d'un vecteur propre l'egalite
`A x = lambda x` : `A` n'en change pas la direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


def combinaison_lineaire(
    coefficients: Sequence[object],
    vecteurs: Sequence[sp.Matrix],
) -> sp.Matrix:
    """`somme(k_i x_i)`, les vecteurs etant de meme dimension."""
    if len(coefficients) != len(vecteurs):
        msg = f"{len(coefficients)} coefficients pour {len(vecteurs)} vecteurs"
        raise ValueError(msg)
    dimensions = {v.shape for v in vecteurs}
    if len(dimensions) > 1:
        msg = f"les vecteurs ne sont pas du meme espace : {sorted(dimensions)}"
        raise ValueError(msg)
    somme = sp.zeros(*vecteurs[0].shape)
    for coefficient, vecteur in zip(coefficients, vecteurs, strict=True):
        somme += sp.sympify(coefficient) * vecteur
    return somme


def sont_colineaires(premier: sp.Matrix, second: sp.Matrix) -> bool:
    """Vrai si deux vecteurs **non nuls** portent la meme direction."""
    if premier.is_zero_matrix or second.is_zero_matrix:
        return False
    return bool(premier.row_join(second).rank() == 1)
