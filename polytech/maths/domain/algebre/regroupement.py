"""La factorisation par regroupement, « un cas particulier » du syllabus (IV.2 4).

On partage le polynome en groupes de termes ; chaque groupe se factorise,
et un facteur commun a tous les groupes se met en evidence :
`(3x^3 - 3) + (7x^2 - 7x) = 3(x - 1)(x^2 + x + 1) + 7x(x - 1)
= (x - 1)(3x^2 + 10x + 3)` (page 51).
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence


def par_regroupement(
    groupes: Sequence[sp.Expr], variable: sp.Symbol = VARIABLE
) -> tuple[sp.Expr, sp.Expr]:
    """Le facteur commun aux groupes, et la somme de ce qu'il en reste."""
    if len(groupes) < 2:  # noqa: PLR2004 -- regrouper, c'est au moins deux
        msg = "un regroupement demande au moins deux groupes"
        raise ValueError(msg)
    commun = sp.factor(reduce(sp.gcd, (sp.expand(g) for g in groupes)))
    if not sp.sympify(commun).has(variable):
        msg = f"aucun facteur commun en {variable} entre les groupes {groupes}"
        raise ValueError(msg)
    cofacteur = sp.expand(sum(sp.cancel(g / commun) for g in groupes))
    return commun, cofacteur
