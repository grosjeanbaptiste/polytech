"""Les equations electromagnetiques d'un systeme de circuits.

Le syllabus d'electricite generale (9.2 a 9.11), miroir magnetique des
paragraphes 3.9 a 3.13 : pour `n` circuits parcourus par des courants `I_k`
et traverses par des flux `phi_k`,

    phi_i = somme des L_ik I_k,      soit  Phi = L I               (9.6.2), (9.6.4)
    W_m   = (1/2) somme I_k phi_k = (1/2) somme L_ik I_i I_k       (9.5.4), (9.6.9)

L'identite de Gauss `somme I'_k phi_k = somme I_k phi'_k` (9.4.1) rend la
matrice `L` symetrique (9.6.6). Une inductance propre est toujours positive
ou nulle, une inductance mutuelle peut etre negative, et elle est bornee :
`|L_12| <= (L_1 + L_2) / 2` (9.9.8), sans quoi deux circuits en serie
discordante auraient une inductance negative.

Le flux d'une bobine de `N` spires traversees par le meme flux `Phi` est le
flux total `phi = N Phi` (9.11.1) : c'est lui qui vaut `L I`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SystemeDeCircuits:
    """Un systeme de circuits, par sa matrice `L` des inductances."""

    inductances: sp.Matrix

    def __post_init__(self) -> None:
        """Refuse une matrice qui ne peut pas etre celle d'un systeme reel."""
        matrice = self.inductances
        if sp.simplify(matrice - matrice.T) != sp.zeros(*matrice.shape):
            msg = "inductances non symetriques : l'identite de Gauss (9.4.1) est violee"
            raise ValueError(msg)
        for rang in range(matrice.rows):
            if matrice[rang, rang].is_negative:
                propre = matrice[rang, rang]
                msg = f"inductance propre negative au rang {rang + 1} : {propre}"
                raise ValueError(msg)
        for i, j in combinations(range(matrice.rows), 2):
            exces = sp.Abs(matrice[i, j]) - (matrice[i, i] + matrice[j, j]) / 2
            if exces.is_positive:
                couple = f"entre {i + 1} et {j + 1}"
                msg = f"inductance mutuelle trop grande {couple} : {matrice[i, j]}"
                raise ValueError(msg)

    def flux(self, courants: Sequence[Any] | sp.Matrix) -> sp.Matrix:
        """`Phi = L I` : les flux a travers chaque circuit."""
        return self.inductances * sp.Matrix(list(courants))

    def energie(self, courants: Sequence[Any] | sp.Matrix) -> sp.Expr:
        """`W_m = (1/2) somme I_k phi_k`."""
        colonne = sp.Matrix(list(courants))
        return colonne.dot(self.flux(colonne)) / 2


def flux_d_un_circuit_de_section_finie(tubes: Sequence[tuple[Any, Any]]) -> sp.Expr:
    """`phi = (1 / I) somme des phi_0 dI`, sur des tubes `(flux, courant)`."""
    total = sp.Add(*(sp.sympify(courant) for _, courant in tubes))
    if total == 0:
        msg = "courant total nul : la moyenne ponderee des flux n'est pas definie"
        raise ValueError(msg)
    return sp.Add(*(sp.sympify(flux) * courant for flux, courant in tubes)) / total


def flux_total(*, spires: int, par_spire: float) -> float:
    """`phi = N Phi` : le flux d'enroulement, et non le flux par spire."""
    if spires < 1:
        msg = f"nombre de spires strictement positif attendu, et non {spires}"
        raise ValueError(msg)
    return spires * par_spire


def flux_a_travers_des_aires(induction: Any, aires: Sequence[tuple[int, Any]]) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`B` fois la somme des aires, chacune comptee autant de fois qu'enlacee."""
    return induction * sum(tours * aire for tours, aire in aires)
