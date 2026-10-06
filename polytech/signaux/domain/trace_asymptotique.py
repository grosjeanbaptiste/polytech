"""Mettre une transmittance sous forme de Bode, pour en tracer les asymptotes.

Planches « Analyse harmonique », « Courbes de Bode - trace asymptotique » :
le travail part de la forme factorisee

`K p^(-alpha) produit(1 + p/z_i) / produit(1 + p/p_j)`.

`alpha` est la classe -- negative si la transmittance derive --, `K` le
gain de Bode, et chaque racine non nulle donne une cassure a son module :
montante pour un zero, descendante pour un pole. Une paire complexe a
deux racines de meme module, sa pulsation propre : elle casse deux fois
au meme endroit.
"""

from __future__ import annotations

import sympy as sp

from polytech.signaux.domain.bode import TraceAsymptotique
from polytech.signaux.domain.symboles import IMAGE as P


def _racines(polynome: sp.Expr) -> dict[sp.Expr, int]:
    """Les racines et leur multiplicite."""
    racines: dict[sp.Expr, int] = sp.roots(sp.Poly(polynome, P))
    return racines


def _cassures(racines: dict[sp.Expr, int], saut: int) -> list[tuple[float, int]]:
    return [
        (float(sp.Abs(racine)), saut)
        for racine, multiplicite in racines.items()
        if racine != 0
        for _ in range(multiplicite)
    ]


def trace_d_une_transmittance(transmittance: sp.Expr) -> TraceAsymptotique:
    """Le gain de Bode, la classe et les cassures de cette fraction rationnelle."""
    numerateur, denominateur = sp.fraction(sp.cancel(transmittance))
    zeros, poles = _racines(numerateur), _racines(denominateur)
    classe = poles.get(sp.S.Zero, 0) - zeros.get(sp.S.Zero, 0)
    gain = sp.limit(sp.cancel(transmittance * P**classe), P, 0)
    if not gain.is_positive:
        msg = f"gain de Bode non positif : {gain} -- le module n'a pas de logarithme"
        raise ValueError(msg)
    cassures = sorted(_cassures(zeros, 1) + _cassures(poles, -1))
    return TraceAsymptotique(
        gain=float(gain), classe=int(classe), cassures=tuple(cassures)
    )
