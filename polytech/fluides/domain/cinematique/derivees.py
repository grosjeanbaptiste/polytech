"""Derivees partielles d'un champ plan (cours III.2).

Le corrige les affiche une a une avant d'en deduire divergence, vorticite et
acceleration : elles sont donc un resultat a part entiere, pas un detail
d'implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y


@dataclass(frozen=True, slots=True)
class DeriveesPartielles:
    """Les six derivees partielles de ``(u, v)`` par rapport a x, y et t."""

    du_dt: sp.Expr
    du_dx: sp.Expr
    du_dy: sp.Expr
    dv_dt: sp.Expr
    dv_dx: sp.Expr
    dv_dy: sp.Expr


def derivees_partielles(u: sp.Expr, v: sp.Expr) -> DeriveesPartielles:
    """Calcule les six derivees partielles du champ."""
    return DeriveesPartielles(
        du_dt=sp.simplify(sp.diff(u, T)),
        du_dx=sp.simplify(sp.diff(u, X)),
        du_dy=sp.simplify(sp.diff(u, Y)),
        dv_dt=sp.simplify(sp.diff(v, T)),
        dv_dx=sp.simplify(sp.diff(v, X)),
        dv_dy=sp.simplify(sp.diff(v, Y)),
    )
