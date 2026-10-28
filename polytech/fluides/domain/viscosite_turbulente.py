"""L'hypothese de Boussinesq de la turbulence : une viscosite turbulente.

Le tenseur de Reynolds `R_ij = -rho <u_i' u_j'>` est inconnu (probleme de
fermeture). Boussinesq le modelise sur la contrainte visqueuse :
`R_ij = mu_t (d_j u_i + d_i u_j) - (2/3) rho k delta_ij`, ou le second
terme assure `R_kk = -2 rho k` quand `div V = 0`. A ne pas confondre avec
l'approximation de Boussinesq de la flottabilite (`boussinesq.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

Matrice = tuple[tuple[float, ...], ...]
DIMENSION = 3


def tenseur_de_boussinesq(
    viscosite_turbulente: float,
    gradient: Sequence[Sequence[float]],
    masse_volumique: float,
    energie_cinetique: float,
) -> Matrice:
    """`R_ij` ; `gradient[i][j] = d u_i / d x_j`, matrice 3 x 3."""
    if viscosite_turbulente < 0:
        msg = f"viscosite turbulente negative : {viscosite_turbulente}"
        raise ValueError(msg)
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)
    if energie_cinetique < 0:
        msg = f"energie cinetique negative : {energie_cinetique} m^2/s^2"
        raise ValueError(msg)
    if len(gradient) != DIMENSION or any(len(ligne) != DIMENSION for ligne in gradient):
        msg = "le gradient de vitesse doit etre une matrice 3 x 3"
        raise ValueError(msg)
    pression = 2 / 3 * masse_volumique * energie_cinetique
    return tuple(
        tuple(
            viscosite_turbulente * (gradient[i][j] + gradient[j][i])
            - (pression if i == j else 0.0)
            for j in range(DIMENSION)
        )
        for i in range(DIMENSION)
    )
