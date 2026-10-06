"""TP « Reservoir », premiere partie : le modele, son repos, sa linearisation.

Trois cuves parfaitement melangees en serie. Un debit principal `F_B`,
de concentration `x_B`, et un debit auxiliaire de produit pur `F_A`, que
regle une vanne d'ouverture `nu`. Les concentrations sont en pour cent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numpy.typing import NDArray

VOLUMES: Final = (35.0, 52.0, 17.0)
"""`V_1`, `V_2`, `V_3`, en m^3."""
RANGEABILITE: Final = 50.0
DEBIT_MAXIMAL: Final = 0.00612
"""`F_Amax`, en m^3/s."""
DEBIT_PRINCIPAL: Final = 0.115
"""`F_B`, en m^3/s."""
PRODUIT_PUR: Final = 100.0
OUVERTURE_DE_REPOS: Final = 0.75
ALIMENTATION_DE_REPOS: Final = 5.0
ECHELON_D_OUVERTURE: Final = 0.1
ECHELON_D_ALIMENTATION: Final = -0.5


def debit_auxiliaire(ouverture: float) -> float:
    """`F_A = F_Amax R^(nu - 1)` : la vanne, non lineaire."""
    return DEBIT_MAXIMAL * math.pow(RANGEABILITE, ouverture - 1)


def entree_des_cuves(debit: float, alimentation: float) -> float:
    """`x_0`, la concentration du melange des deux debits."""
    total = debit + DEBIT_PRINCIPAL
    return (DEBIT_PRINCIPAL * alimentation + debit * PRODUIT_PUR) / total


def concentration_de_repos() -> float:
    """Au repos les trois cuves ont la concentration de leur alimentation."""
    return entree_des_cuves(debit_auxiliaire(OUVERTURE_DE_REPOS), ALIMENTATION_DE_REPOS)


def gain_de_la_vanne() -> float:
    """`dF_A/dnu` au repos : `F_A ln R`."""
    nu = sp.Symbol("nu")
    vanne = DEBIT_MAXIMAL * sp.Float(RANGEABILITE) ** (nu - 1)
    return float(sp.diff(vanne, nu).subs(nu, OUVERTURE_DE_REPOS))


def gains_du_melange() -> tuple[float, float]:
    """`dx_0/dF_A` et `dx_0/dx_B` au repos."""
    debit, alimentation = sp.symbols("F_A x_B")
    total = debit + DEBIT_PRINCIPAL
    melange = (DEBIT_PRINCIPAL * alimentation + debit * PRODUIT_PUR) / total
    repos = {
        debit: debit_auxiliaire(OUVERTURE_DE_REPOS),
        alimentation: ALIMENTATION_DE_REPOS,
    }
    return (
        float(sp.diff(melange, debit).subs(repos)),
        float(sp.diff(melange, alimentation).subs(repos)),
    )


def constantes_de_temps() -> tuple[float, float, float]:
    """`V_i/(F_A + F_B)` : le temps de sejour dans chaque cuve."""
    total = debit_auxiliaire(OUVERTURE_DE_REPOS) + DEBIT_PRINCIPAL
    premiere, deuxieme, troisieme = (volume / total for volume in VOLUMES)
    return premiere, deuxieme, troisieme


def variations_finales_lineaires() -> tuple[float, float]:
    """La variation finale de `x_3` pour chacun des deux echelons de l'enonce."""
    par_le_debit, par_la_concentration = gains_du_melange()
    commande = gain_de_la_vanne() * par_le_debit * ECHELON_D_OUVERTURE
    return commande, par_la_concentration * ECHELON_D_ALIMENTATION


def simuler(
    *, ouverture: float, alimentation: float, duree: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Les trois concentrations, par le modele non lineaire, depuis le repos."""
    debit = debit_auxiliaire(ouverture)
    entree = entree_des_cuves(debit, alimentation)

    def bilans(_: float, x: NDArray[np.float64]) -> list[float]:
        amont = (entree, x[0], x[1])
        return [
            (debit + DEBIT_PRINCIPAL) * (a - c) / v
            for a, c, v in zip(amont, x, VOLUMES, strict=True)
        ]

    instants = np.linspace(0.0, duree, 601)
    depart = [concentration_de_repos()] * 3
    solution = solve_ivp(bilans, (0.0, duree), depart, t_eval=instants, rtol=1e-9)
    return instants, np.asarray(solution.y.T, dtype=float)
