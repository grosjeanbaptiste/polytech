"""Le rayon de convergence quand `|a_n|^(1/n)` n'a pas de limite.

« Il n'est pas necessaire que la limite existe pour pouvoir conclure »
(Mathematiques appliquees, remarque 4.1) : la **limite superieure**
`alpha = lim sup |a_n|^(1/n)` existe toujours, et `R = 1 / alpha`.

C'est le cas des series dont des coefficients sont nuls -- `cosh x`,
`sin x` -- et de celles dont les coefficients alternent entre deux lois.
La suite se decoupe alors en `p` sous-suites `a_(pm + r)`, une par reste
`r`, qui ont chacune une limite ; la limite superieure est la plus grande.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.series.entieres import INDICE

if TYPE_CHECKING:
    from collections.abc import Sequence


def _limite_de_la_racine(terme: sp.Expr, pas: int, reste: int) -> sp.Expr:
    """`lim |a_(pm + r)|^(1 / (pm + r))` quand `m` tend vers l'infini."""
    rang = pas * INDICE + reste
    return sp.limit(sp.Abs(terme) ** (1 / rang), INDICE, sp.oo)


def rayon_par_la_limite_superieure(sous_suites: Sequence[object]) -> sp.Expr:
    """`R = 1 / lim sup |a_n|^(1/n)`, la suite etant donnee reste par reste.

    `sous_suites[r]` est `a_(pm + r)` ecrit en `m` (le symbole `INDICE`),
    pour `p = len(sous_suites)` ; une sous-suite nulle s'ecrit `0`.
    """
    termes = [sp.sympify(terme) for terme in sous_suites]
    if not termes:
        msg = "aucune sous-suite : la suite des coefficients n'est pas donnee"
        raise ValueError(msg)
    limites = [
        _limite_de_la_racine(terme, len(termes), reste)
        for reste, terme in enumerate(termes)
        if terme != 0
    ]
    if not limites:
        msg = "tous les coefficients sont nuls : la serie est la fonction nulle"
        raise ValueError(msg)
    superieure = sp.Max(*limites)
    if superieure == 0:
        return sp.oo
    return sp.Integer(0) if superieure == sp.oo else 1 / superieure
