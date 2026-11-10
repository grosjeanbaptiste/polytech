"""Le probleme aux limites du second ordre : `y'' = f(x, y, y')`.

Contrairement au probleme de Cauchy, les deux conditions sont posees en deux
points : `y(a) = ya`, `y(b) = yb`. L'existence et l'unicite ne sont plus
garanties — `y'' + y = 0`, `y(0) = 0`, `y(pi) = 1` n'a aucune solution — et
la resolution est numerique (`solve_bvp`) : un echec de convergence leve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_bvp

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

POINTS_DE_DEPART = 11
"""Le maillage initial de `solve_bvp`, qu'il raffine."""


@dataclass(frozen=True, slots=True)
class Bords:
    """`y(a) = ya` et `y(b) = yb`."""

    a: float
    b: float
    ya: float
    yb: float


def resoudre_aux_limites(
    second_membre: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.float64],
    ],
    bords: Bords,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """`y(x)` tel que `y'' = f(x, y, y')` et `y` prend les valeurs des bords."""
    if not bords.a < bords.b:
        msg = f"a < b requis, recu a={bords.a}, b={bords.b}"
        raise ValueError(msg)

    def systeme(x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack([v[1], second_membre(x, v[0], v[1])])

    def conditions(
        gauche: NDArray[np.float64], droite: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.array([gauche[0] - bords.ya, droite[0] - bords.yb])

    maillage = np.linspace(bords.a, bords.b, POINTS_DE_DEPART)
    depart = np.vstack(
        [
            np.interp(maillage, [bords.a, bords.b], [bords.ya, bords.yb]),
            np.zeros_like(maillage),
        ]
    )
    solution = solve_bvp(systeme, conditions, maillage, depart, tol=1e-9)
    if not solution.success:
        msg = f"pas de solution trouvee : {solution.message}"
        raise ValueError(msg)

    def y(x: NDArray[np.float64]) -> NDArray[np.float64]:
        valeurs: NDArray[np.float64] = solution.sol(x)[0]
        return valeurs

    return y
