"""TP « Reservoir » : la courbe de Black d'une boucle du troisieme ordre.

L'argument d'un troisieme ordre descend sous -180 degres. Calcule point
par point, il reviendrait a +180 : il faut le **derouler**, c'est-a-dire
suivre la courbe sans jamais sauter de 360 degres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from polytech.signaux.application.formatage import coordonnees
from polytech.signaux.domain.symboles import IMAGE as P

if TYPE_CHECKING:
    from numpy.typing import NDArray


def points_de_black(
    ouverte: sp.Expr, debut: float, fin: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Les arguments deroules, en degres, et les gains, en decibels."""
    pulsations = np.logspace(np.log10(debut), np.log10(fin), 161)
    valeurs = sp.lambdify(P, ouverte, "numpy")(1j * pulsations)
    arguments = np.degrees(np.unwrap(np.angle(valeurs)))
    return np.asarray(arguments, dtype=float), 20 * np.log10(np.abs(valeurs))


def black(ouverte: sp.Expr, debut: float, fin: float) -> str:
    """La courbe de Black, prete a tracer."""
    arguments, gains = points_de_black(ouverte, debut, fin)
    return coordonnees(arguments, gains, 3)
