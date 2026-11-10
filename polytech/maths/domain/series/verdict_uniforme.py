"""Le verdict de convergence uniforme d'une suite de fonctions.

`f_n -> f` uniformement sur un ensemble si et seulement si
`d_n = sup |f_n - f|` tend vers 0. Le cours le pose sur deux suites : `x^n`
sur `[0, a]` (`d_n = a^n`, uniforme) et `x^n` sur `[0, 1[` (`d_n = 1`, pas
uniforme).

La distance `d_n` se donne en forme fermee, comme le cours l'ecrit ; elle
n'est crue qu'apres recoupement avec le plus grand ecart calcule sur
quelques rangs. Une limite que sympy ne tranche pas leve : un verdict
n'est jamais devine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.series.suites_de_fonctions import distances_uniformes

if TYPE_CHECKING:
    from collections.abc import Sequence


class Verdict(Enum):
    """Ce que la limite de `d_n` dit de la convergence."""

    UNIFORME = "uniforme"
    PAS_UNIFORME = "pas uniforme"


@dataclass(frozen=True, slots=True)
class Suite:
    """`f_n(x)`, sa limite simple, l'indice `n` et l'ensemble des `x`."""

    terme: sp.Expr
    limite: object
    indice: sp.Symbol
    intervalle: sp.Set


def _recouper(suite: Suite, distance: sp.Expr, rangs: Sequence[int]) -> None:
    """Leve si `distance` ne vaut pas le plus grand ecart sur un des rangs."""
    calculees = distances_uniformes(
        sp.sympify(suite.terme), suite.limite, suite.indice, suite.intervalle, rangs
    )
    for rang, calculee in zip(rangs, calculees, strict=True):
        annoncee = distance.subs(suite.indice, rang)
        if sp.simplify(annoncee - calculee) != 0:
            msg = (
                f"d_{rang} annoncee {annoncee} ne vaut pas "
                f"le plus grand ecart {calculee}"
            )
            raise ValueError(msg)


def verdict(suite: Suite, distance: sp.Expr, rangs: Sequence[int]) -> Verdict:
    """`UNIFORME` si `d_n -> 0`, `PAS_UNIFORME` sinon ; leve si indeterminee."""
    _recouper(suite, distance, rangs)
    tendue = sp.limit(distance, suite.indice, sp.oo)
    if tendue.has(sp.AccumBounds) or not tendue.is_number:
        msg = f"limite de la distance indeterminee : {tendue}"
        raise ValueError(msg)
    return Verdict.UNIFORME if tendue == 0 else Verdict.PAS_UNIFORME
