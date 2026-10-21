"""Le changement `t = tg x`, pour une fonction paire en `sin x` et `cos x`.

`regles_de_parite.changement_conseille` dit quand le conseiller ; ce
module le fait, des que `R(-sin x, -cos x) = R`, c'est-a-dire
`f(x + pi) = f(x)` -- meme si une regle plus simple s'applique aussi.
Une fonction paire ne garde que `cos^2 x = 1 / (1 + t^2)`,
`sin^2 x = t^2 / (1 + t^2)` et `sin x cos x = t / (1 + t^2)`, et
`dx = dt / (1 + t^2)` (notes de 2011, pages 53 et 54).
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE

TANGENTE: Final = sp.Symbol("t", real=True)
"""`t = tg x`."""


def en_tangente(integrande: sp.Expr) -> sp.Expr:
    """L'integrande en `t = tg x`, le facteur `dx = dt / (1 + t^2)` compris."""
    if sp.simplify(integrande.subs(VARIABLE, VARIABLE + sp.pi) - integrande) != 0:
        msg = f"t = tg x demande une fonction paire en sin x et cos x : {integrande}"
        raise ValueError(msg)
    en_t = integrande.subs(VARIABLE, sp.atan(TANGENTE)) / (1 + TANGENTE**2)
    return sp.factor(sp.simplify(en_t))
