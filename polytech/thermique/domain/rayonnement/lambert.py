"""L'exitance, la luminance, et la loi de Planck telle que le cours l'ecrit.

Syllabus de thermique, tome 1, deuxieme partie, sections 1.2 a 1.4.
L'exitance `M` est le flux emis par unite d'aire, toutes directions et
toutes longueurs d'onde confondues ; la luminance `L` le rapporte a
l'angle solide et a la surface projetee. Elles sont liees par

    M = integrale sur l'hemisphere de L cos(alpha) d(omega),

et, pour un corps dont l'emission est isotrope -- lambertien --,
`M = pi L`.

La loi de Planck donne la luminance spectrale du corps noir,
`l = c_1 lambda^-5 / (exp(c_2 / (lambda T)) - 1)`, avec `c_1 = 2 h c^2`
et `c_2 = h c / k`. `corps_noir.luminance_spectrale` rend, malgre son
nom, l'exitance spectrale : `pi` fois celle-ci.

Un corps quelconque emet `l = alpha l_o` (Kirchhoff), et un corps gris
`M = epsilon sigma T^4`.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.thermique.domain.rayonnement.corps_noir import (
    BOLTZMANN,
    LUMIERE,
    PLANCK,
    emittance,
    longueur_d_onde_du_maximum,
)

ALPHA: Final = sp.Symbol("alpha", nonnegative=True)
"""L'angle entre la direction d'emission et la normale."""
PREMIERE_CONSTANTE: Final = 2.0 * PLANCK * LUMIERE**2
"""`c_1 = 2 h c^2`, en W m^2/sr."""
SECONDE_CONSTANTE: Final = PLANCK * LUMIERE / BOLTZMANN
"""`c_2 = h c / k`, en m K."""
EXPOSANT_MAXIMAL: Final = 700.0
"""Au-dela, `exp` deborde le flottant, et la luminance vaut zero."""


def exitance_par_integration(luminance: sp.Expr) -> sp.Expr:
    """`M`, la luminance -- fonction de `ALPHA` -- integree sur l'hemisphere."""
    azimut = sp.Symbol("phi", nonnegative=True)
    element = luminance * sp.cos(ALPHA) * sp.sin(ALPHA)
    return sp.integrate(element, (ALPHA, 0, sp.pi / 2), (azimut, 0, 2 * sp.pi))


def luminance_de_planck(longueur_d_onde: float, temperature: float) -> float:
    """`l_o`, en W/(m^2 m sr)."""
    if longueur_d_onde <= 0:
        msg = f"longueur d'onde nulle ou negative : {longueur_d_onde} m"
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"temperature nulle ou negative : {temperature} K"
        raise ValueError(msg)
    exposant = SECONDE_CONSTANTE / (longueur_d_onde * temperature)
    if exposant > EXPOSANT_MAXIMAL:
        return 0.0  # l'exponentielle deborde le flottant : c'est la limite exacte
    return PREMIERE_CONSTANTE / (longueur_d_onde**5 * math.expm1(exposant))


def luminance_du_corps_noir(temperature: float) -> float:
    """`L_o = sigma T^4 / pi`, en W/(m^2 sr)."""
    return emittance(temperature) / math.pi


def _exiger_un_facteur(facteur: float) -> None:
    if not 0 <= facteur <= 1:
        msg = f"facteur hors de [0, 1] : {facteur}"
        raise ValueError(msg)


def luminance_emise(
    absorption: float, longueur_d_onde: float, temperature: float
) -> float:
    """`l = alpha l_o` : un corps emet ce qu'il absorberait du corps noir."""
    _exiger_un_facteur(absorption)
    return absorption * luminance_de_planck(longueur_d_onde, temperature)


def exitance_grise(emissivite: float, temperature: float) -> float:
    """`M = epsilon sigma T^4`, en W/m^2."""
    _exiger_un_facteur(emissivite)
    return emissivite * emittance(temperature)


def luminance_maximale(temperature: float) -> float:
    """La luminance de Planck a son sommet, `lambda_max T = 2898 um K`."""
    return luminance_de_planck(longueur_d_onde_du_maximum(temperature), temperature)


def angle_solide(aire: float, rayon: float) -> float:
    """`omega = dA / r^2`, en steradians : l'aire interceptee sur la sphere unite."""
    if rayon <= 0:
        msg = f"rayon nul ou negatif : {rayon} m"
        raise ValueError(msg)
    return aire / rayon**2
