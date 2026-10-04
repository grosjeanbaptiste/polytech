"""Chapitre XIV.3.5 b) : les priorites de separation.

Certaines variables jouent dans la modelisation un role plus important
que d'autres -- les variables 0 - 1 qui conditionnent la solution, par
exemple. Les logiciels laissent donc l'utilisateur leur donner des
priorites : au moment de choisir la variable de separation, seules
comptent, parmi les variables fractionnaires, celles de plus haute
priorite. La regle de selection (penalites, non-integralite) departage
ensuite celles-la.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

SANS_PRIORITE = float("-inf")
"""Une variable que l'utilisateur n'a pas classee passe apres toutes les autres."""


def variables_prioritaires(
    fractionnaires: frozenset[int],
    priorites: Mapping[int, int],
) -> frozenset[int]:
    """Parmi les variables fractionnaires, celles de plus haute priorite."""
    if not fractionnaires:
        return frozenset()
    rangs = {v: priorites.get(v, SANS_PRIORITE) for v in fractionnaires}
    plus_haute = max(rangs.values())
    return frozenset(v for v, rang in rangs.items() if rang == plus_haute)
