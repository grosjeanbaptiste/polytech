"""L'ecoulement turbulent : echelles, tenseur de Reynolds, conduite.

Planches du chapitre VIII. L'energie injectee aux grandes echelles
cascade jusqu'a l'echelle de Kolmogorov, `eta = (nu^3 / epsilon)^(1/4)`,
ou la viscosite la dissipe.

Moyenner les equations de Navier-Stokes y laisse un terme de plus, le
tenseur de Reynolds `R_ij = -rho u_i' u_j'` : il est inconnu, et c'est lui
qu'un modele de turbulence doit fournir. L'energie cinetique de
turbulence est la demi-somme des variances, `k = -R_kk / (2 rho)`.

Dans une conduite, la contrainte totale croit lineairement de l'axe a la
paroi. Loin de la paroi, la vitesse suit la loi deficitaire
`(u_max - u) / u* = 2,5 ln(R / (R - r))` : la loi logarithmique, dont la
constante est evaluee sur l'axe.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import sympy as sp

from polytech.fluides.domain.turbulence import KARMAN, covariance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.fluides.domain.turbulence import Decomposition


def echelle_de_kolmogorov(viscosite: float, dissipation: float) -> float:
    """`eta = (nu^3 / epsilon)^(1/4)`, en metres."""
    if dissipation <= 0:
        msg = f"dissipation nulle ou negative : {dissipation} W/kg"
        raise ValueError(msg)
    return float((viscosite**3 / dissipation) ** 0.25)


def tenseur_de_reynolds(
    masse_volumique: float, composantes: Sequence[Decomposition]
) -> sp.Matrix:
    """`R_ij = -rho moyenne(u_i' u_j')`."""
    taille = len(composantes)
    return sp.Matrix(
        taille,
        taille,
        lambda i, j: -masse_volumique * covariance(composantes[i], composantes[j]),
    )


def energie_cinetique_turbulente(composantes: Sequence[Decomposition]) -> float:
    """`k = (u_x'^2 + u_y'^2 + u_z'^2) / 2`, en m^2/s^2."""
    return sum(composante.variance() for composante in composantes) / 2


def _exiger_dans_la_conduite(position: float, rayon: float, *, paroi: bool) -> None:
    dedans = 0 <= position <= rayon if paroi else 0 <= position < rayon
    if not dedans:
        msg = f"position hors de la conduite : r = {position} m pour R = {rayon} m"
        raise ValueError(msg)


def contrainte_reduite_en_conduite(position: float, rayon: float) -> float:
    """`tau / tau_w = 1 - (R - r) / R` : nulle sur l'axe, un a la paroi."""
    _exiger_dans_la_conduite(position, rayon, paroi=True)
    return 1 - (rayon - position) / rayon


def deficit_de_vitesse(position: float, rayon: float) -> float:
    """`(u_max - u) / u* = (1 / kappa) ln(R / (R - r))`."""
    _exiger_dans_la_conduite(position, rayon, paroi=False)
    return math.log(rayon / (rayon - position)) / KARMAN
