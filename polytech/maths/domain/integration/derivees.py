"""La table des derivees que le cours rappelle avant de calculer des primitives.

Planche 33 : quatorze formules, lues a l'envers pour integrer. Elles sont
ici une **donnee**, recopiee de la planche ; un test les retrouve une a
une en derivant, pour qu'une faute de recopie ne passe pas.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X

DERIVEES_USUELLES: Final[dict[sp.Expr, sp.Expr]] = {
    sp.sin(X): sp.cos(X),
    sp.cos(X): -sp.sin(X),
    sp.tan(X): 1 / sp.cos(X) ** 2,
    sp.exp(X): sp.exp(X),
    sp.sinh(X): sp.cosh(X),
    sp.cosh(X): sp.sinh(X),
    sp.tanh(X): 1 / sp.cosh(X) ** 2,
    sp.asin(X): 1 / sp.sqrt(1 - X**2),
    sp.acos(X): -1 / sp.sqrt(1 - X**2),
    sp.atan(X): 1 / (1 + X**2),
    sp.log(X): 1 / X,
    sp.asinh(X): 1 / sp.sqrt(X**2 + 1),
    sp.acosh(X): 1 / sp.sqrt(X**2 - 1),
    sp.atanh(X): 1 / (1 - X**2),
}
"""`f(x)` et sa derivee, dans l'ordre de la planche 33."""


def derivee_usuelle(fonction: sp.Expr) -> sp.Expr:
    """La derivee que la table donne ; leve si la fonction n'y figure pas."""
    if fonction not in DERIVEES_USUELLES:
        msg = f"{fonction} n'est pas dans la table des derivees de la planche 33"
        raise ValueError(msg)
    return DERIVEES_USUELLES[fonction]
