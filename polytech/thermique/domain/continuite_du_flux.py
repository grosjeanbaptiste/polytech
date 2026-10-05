"""La temperature de paroi que la continuite du flux impose.

Syllabus de thermique, tome 2, deuxieme partie, section 1.5. Un
coefficient de transmission depend le plus souvent de la temperature de
la paroi qu'il baigne -- par le nombre de Grashof en convection
naturelle, par `T^4` en rayonnement, par la surchauffe en ebullition.
Cette temperature n'est pas connue d'avance : elle resulte des memes
coefficients.

On ecrit donc que le flux qui arrive a la surface est celui qui en
repart, `F_1(t_p) = F_2(t_p)` -- l'equation (1.13) de la tuyauterie
calorifugee. `F_1` decroit avec `t_p`, `F_2` croit : la figure 1.11 les
montre se croiser en un seul point. Le syllabus nomme la « regula
falsi » pour le trouver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.maths.domain.fausse_position import racine_par_fausse_position

if TYPE_CHECKING:
    from collections.abc import Callable

TOLERANCE = 1e-9
"""En W/m^2 : bien en deca de ce qu'une correlation garantit."""


def temperature_superficielle(
    entrant: Callable[[float], float],
    sortant: Callable[[float], float],
    bornes: tuple[float, float],
) -> float:
    """`t_p` telle que `entrant(t_p) = sortant(t_p)`, entre les deux `bornes`."""

    def desequilibre(paroi: float) -> float:
        return entrant(paroi) - sortant(paroi)

    return racine_par_fausse_position(desequilibre, *bornes, tolerance=TOLERANCE)
