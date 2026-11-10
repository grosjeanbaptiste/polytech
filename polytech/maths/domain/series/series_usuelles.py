"""Les six series usuelles du formulaire (section 8, lignes 577-618).

`1/(1-x)` pour `|x| < 1`, puis `e^x`, `cosh`, `sinh`, `cos` et `sin` pour
tout `x` reel. Chaque serie se donne reste par reste, comme le fait la
limite superieure du rayon : `sous_suites[r]` est `a_(pm + r)` ecrit en `m`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.series.entieres import INDICE
from polytech.maths.domain.series.lacunaires import rayon_par_la_limite_superieure
from polytech.maths.domain.symboles import VARIABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

_PAIR: Final = sp.factorial(2 * INDICE)
_IMPAIR: Final = sp.factorial(2 * INDICE + 1)


@dataclass(frozen=True, slots=True)
class SerieUsuelle:
    """La fonction, et ses coefficients donnes reste par reste."""

    fonction: sp.Expr
    sous_suites: Sequence[sp.Expr]

    def coefficient(self, rang: int) -> sp.Expr:
        """`a_rang`."""
        pas = len(self.sous_suites)
        terme = self.sous_suites[rang % pas]
        return sp.sympify(terme).subs(INDICE, rang // pas)

    def somme_partielle(self, point: object, ordre: int) -> sp.Expr:
        """`somme a_n x^n`, jusqu'a `n = ordre` compris, en `x = point`."""
        return sum(
            (self.coefficient(n) * sp.sympify(point) ** n for n in range(ordre + 1)),
            sp.Integer(0),
        )

    def rayon(self) -> sp.Expr:
        """`R = 1 / lim sup |a_n|^(1/n)`."""
        return rayon_par_la_limite_superieure(self.sous_suites)


SERIES_USUELLES: Final[dict[str, SerieUsuelle]] = {
    "geometrique": SerieUsuelle(1 / (1 - VARIABLE), (sp.Integer(1),)),
    "exponentielle": SerieUsuelle(sp.exp(VARIABLE), (1 / sp.factorial(INDICE),)),
    "cosh": SerieUsuelle(sp.cosh(VARIABLE), (1 / _PAIR, sp.Integer(0))),
    "sinh": SerieUsuelle(sp.sinh(VARIABLE), (sp.Integer(0), 1 / _IMPAIR)),
    "cos": SerieUsuelle(sp.cos(VARIABLE), ((-1) ** INDICE / _PAIR, sp.Integer(0))),
    "sin": SerieUsuelle(sp.sin(VARIABLE), (sp.Integer(0), (-1) ** INDICE / _IMPAIR)),
}


def serie(nom: str) -> SerieUsuelle:
    """La serie du formulaire de ce nom ; leve `KeyError` si elle n'y est pas."""
    if nom not in SERIES_USUELLES:
        msg = f"serie usuelle inconnue : {nom}"
        raise KeyError(msg)
    return SERIES_USUELLES[nom]
