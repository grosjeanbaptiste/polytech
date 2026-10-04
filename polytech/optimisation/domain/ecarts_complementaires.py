"""Chapitre IV.2.3 : le theoreme des ecarts complementaires.

`x` et `u` admissibles sont optimales si et seulement si

    u (T x - d) = 0        et        (c - u T) x = 0.

Le cours le redit en **couples de contraintes duales** : chaque variable
`x_j >= 0` fait couple avec la contrainte duale `j`, chaque contrainte
primale `i` avec `u_i >= 0`. Une contrainte est **serree** quand elle est
verifiee avec le signe d'egalite ; a l'optimum, tout couple en porte au
moins une.

D'ou un moyen de calcul (page 99) : connaissant la solution primale
optimale, les ecarts donnent un systeme lineaire dont la solution duale
est l'unique solution -- si la primale n'est pas degeneree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.dual import dual_de, sous_la_convention

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.formulation import ProgrammeGeneral


@dataclass(frozen=True, slots=True)
class CoupleDeContraintes:
    """Une contrainte de chaque probleme, et l'ecart de chacune a l'egalite."""

    primale: str
    ecart_primal: sp.Expr
    duale: str
    ecart_dual: sp.Expr

    @property
    def a_une_contrainte_serree(self) -> bool:
        """Vrai si l'un des deux ecarts au moins est nul."""
        return bool(self.ecart_primal * self.ecart_dual == 0)

    @property
    def a_exactement_une_contrainte_serree(self) -> bool:
        """Vrai si un ecart est nul et l'autre non : le cas non degenere."""
        return bool(self.ecart_primal == 0) != bool(self.ecart_dual == 0)


def couples_de_contraintes(
    primal: ProgrammeGeneral,
    x: Sequence[object],
    u: Sequence[object],
) -> tuple[CoupleDeContraintes, ...]:
    """Les `n + m` couples : les variables primales, puis les contraintes reelles."""
    convention = sous_la_convention(primal)
    dual = dual_de(primal)
    variables = tuple(
        CoupleDeContraintes(
            f"x_{j} >= 0",
            sp.sympify(valeur),
            f"contrainte duale {j}",
            dual.contraintes[j - 1].ecart(u),
        )
        for j, valeur in enumerate(x, start=1)
    )
    contraintes = tuple(
        CoupleDeContraintes(
            f"contrainte primale {i}",
            convention.contraintes[i - 1].ecart(x),
            f"u_{i} >= 0",
            sp.sympify(valeur),
        )
        for i, valeur in enumerate(u, start=1)
    )
    return variables + contraintes


def verifie_les_ecarts_complementaires(
    primal: ProgrammeGeneral,
    x: Sequence[object],
    u: Sequence[object],
) -> bool:
    """Vrai si chaque couple porte au moins une contrainte serree."""
    return all(c.a_une_contrainte_serree for c in couples_de_contraintes(primal, x, u))


def duale_par_les_ecarts(
    primal: ProgrammeGeneral,
    x: Sequence[object],
) -> tuple[sp.Expr, ...]:
    """La solution duale optimale, deduite de la primale optimale `x`."""
    inconnues = sp.symbols(f"u1:{len(primal.contraintes) + 1}")
    couples = couples_de_contraintes(primal, x, inconnues)
    equations = [c.ecart_dual if c.ecart_primal != 0 else None for c in couples]
    solutions = sp.linsolve([e for e in equations if e is not None], inconnues)
    (solution,) = tuple(solutions) or (None,)
    if solution is None or any(v.free_symbols for v in solution):
        msg = "les ecarts complementaires ne determinent pas la solution duale"
        raise ValueError(msg)
    return tuple(solution)
