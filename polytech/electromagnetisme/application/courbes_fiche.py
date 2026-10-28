"""Les courbes de la fiche d'electromagnetisme, calculees par la bibliotheque."""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.transitoires import (
    reponse_du_premier_ordre,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

type Points = list[tuple[float, float]]


def _courbe(expression: sp.Expr, debut: float, fin: float, pas: int) -> Points:
    return [
        (
            debut + (fin - debut) * i / pas,
            float(expression.subs(TEMPS, debut + (fin - debut) * i / pas)),
        )
        for i in range(pas + 1)
    ]


def charge() -> Points:
    """`u = u_inf (1 - e^(-t/tau))`, de 0 a 5 `tau` (`tau = 1`, `u_inf = 1`)."""
    return _courbe(reponse_du_premier_ordre(sp.S.Zero, sp.S.One, sp.S.One), 0, 5, 100)


def decharge() -> Points:
    """`u = u_0 e^(-t/tau)`."""
    return _courbe(reponse_du_premier_ordre(sp.S.One, sp.S.Zero, sp.S.One), 0, 5, 100)


def tangente_a_l_origine() -> Points:
    """La tangente de la charge en `t = 0`, de 0 a 1,2 `tau`."""
    pente = sp.diff(reponse_du_premier_ordre(sp.S.Zero, sp.S.One, sp.S.One), TEMPS)
    droite = pente.subs(TEMPS, 0) * TEMPS
    return _courbe(droite, 0, 1.2, 12)


def charge_a_tau() -> float:
    """`1 - e^(-1) = 0,632` : l'ordonnee des 63 % de la figure."""
    return charge()[20][1]
