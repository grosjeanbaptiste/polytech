"""Le vocabulaire d'une contrainte de programmation lineaire (chapitre I.1).

Separe de `formulation`, qui porte les passages d'une forme a l'autre :
ici, ce qu'est une contrainte et dans quel sens on optimise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


class Sens(Enum):
    """Le sens d'une contrainte reelle."""

    INFERIEUR = "<="
    SUPERIEUR = ">="
    EGAL = "="


class Optimisation(Enum):
    """Ce que l'on cherche : le minimum ou le maximum de `z`."""

    MIN = "min"
    MAX = "max"


def exacts(valeurs: Sequence[object]) -> tuple[sp.Expr, ...]:
    """Des rationnels exacts : un simplexe en flottants derive."""
    return tuple(sp.nsimplify(v, rational=True) for v in valeurs)


@dataclass(frozen=True, slots=True)
class Contrainte:
    """`t_i x  (<=, >=, =)  d_i`."""

    coefficients: tuple[sp.Expr, ...]
    sens: Sens
    second_membre: sp.Expr

    def __post_init__(self) -> None:
        """Rend les nombres exacts : un simplexe en flottants derive."""
        object.__setattr__(self, "coefficients", exacts(self.coefficients))
        object.__setattr__(
            self, "second_membre", sp.nsimplify(self.second_membre, rational=True)
        )

    def premier_membre(self, point: Sequence[object]) -> sp.Expr:
        """`t_i x` au point donne."""
        return sp.Add(
            *(a * sp.sympify(x) for a, x in zip(self.coefficients, point, strict=True))
        )

    def est_satisfaite(self, point: Sequence[object]) -> bool:
        """Vrai si le point respecte la contrainte."""
        ecart = self.premier_membre(point) - self.second_membre
        if self.sens is Sens.INFERIEUR:
            return bool(ecart <= 0)
        if self.sens is Sens.SUPERIEUR:
            return bool(ecart >= 0)
        return bool(ecart == 0)

    def ecart(self, point: Sequence[object]) -> sp.Expr:
        """La variable d'ecart : `d - t x` pour `<=`, `t x - d` sinon."""
        difference = self.premier_membre(point) - self.second_membre
        return -difference if self.sens is Sens.INFERIEUR else difference

    def retournee(self) -> Contrainte:
        """La meme contrainte multipliee par `-1` : le sens s'inverse."""
        inverse = {Sens.INFERIEUR: Sens.SUPERIEUR, Sens.SUPERIEUR: Sens.INFERIEUR}
        return Contrainte(
            tuple(-a for a in self.coefficients),
            inverse.get(self.sens, Sens.EGAL),
            -self.second_membre,
        )
