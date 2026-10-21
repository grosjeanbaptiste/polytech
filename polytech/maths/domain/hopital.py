"""La regle de L'Hopital, une application apres l'autre.

Une forme `0/0` ou `oo/oo` en `a` a la meme limite que `f'/g'`, si
celle-ci existe. Chaque quotient est simplifie avant d'etre juge, comme sur la planche,
ou `(1/delta) / (-1/delta^2)` devient `-delta`. On derive tant que la
forme reste indeterminee
(planche 40 du premier fascicule : `lim delta ln delta = 0`). La limite
est prise du cote droit, celui ou la planche fait tendre `delta`.
"""

from __future__ import annotations

import sympy as sp

MAXIMUM = 8


def _est_indeterminee(
    f: sp.Expr, g: sp.Expr, variable: sp.Symbol, point: object
) -> bool:
    lf = sp.limit(f, variable, point, "+")
    lg = sp.limit(g, variable, point, "+")
    return bool((lf == 0 and lg == 0) or (lf.is_infinite and lg.is_infinite))


def par_l_hopital(
    numerateur: sp.Expr,
    denominateur: sp.Expr,
    variable: sp.Symbol,
    point: object,
    *,
    maximum: int = MAXIMUM,
) -> tuple[tuple[sp.Expr, ...], sp.Expr]:
    """Les quotients successifs `f/g, f'/g', ...`, et la limite atteinte."""
    f, g = numerateur, denominateur
    if not _est_indeterminee(f, g, variable, point):
        msg = (
            f"{f} / {g} n'est pas indeterminee en {point} : la regle ne s'applique pas"
        )
        raise ValueError(msg)
    etapes = [f / g]
    while len(etapes) <= maximum:
        quotient = sp.simplify(sp.diff(f, variable) / sp.diff(g, variable))
        etapes.append(quotient)
        f, g = sp.fraction(quotient)
        if not _est_indeterminee(f, g, variable, point):
            return tuple(etapes), sp.limit(sp.simplify(f / g), variable, point, "+")
    msg = f"la regle ne conclut pas en {maximum} applications : {etapes[-1]}"
    raise ValueError(msg)
