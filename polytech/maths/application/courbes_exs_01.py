"""Mathematique, exercices 1 : les courbes de la figure de l'exercice 12.

`arctan(tan(x/2) + 1)` est une primitive sur chaque intervalle
`](2k - 1) pi, (2k + 1) pi[`, et saute de `pi` d'un intervalle au suivant.
La figure montre trois de ces branches, et la primitive continue qui les
relie.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import sympy as sp

from polytech.maths.application.primitives_b_exs_01 import (
    exercice_12,
    primitive_du_12_sur_r,
)
from polytech.maths.domain.symboles import VARIABLE as X

POINTS: Final = 61
MARGE: Final = 0.02
"""L'ecart laisse aux bords d'une branche, ou la tangente diverge."""
Courbe = tuple[np.ndarray, np.ndarray]


def courbes_du_12() -> tuple[tuple[Courbe, ...], Courbe]:
    """Les branches sur trois periodes, puis la primitive continue."""
    par_branche = sp.lambdify(X, exercice_12().primitive, "numpy")
    partout = sp.lambdify(X, primitive_du_12_sur_r(), "numpy")
    branches = []
    for rang in (-1, 0, 1):
        centre = 2 * math.pi * rang
        abscisses = np.linspace(
            centre - math.pi + MARGE, centre + math.pi - MARGE, POINTS
        )
        branches.append((abscisses, par_branche(abscisses)))
    etendue = np.linspace(-3 * math.pi, 3 * math.pi, 3 * POINTS)
    return tuple(branches), (etendue, partout(etendue))
