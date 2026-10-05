"""L'induction d'une boucle circulaire parcourue par un courant.

Le syllabus d'electricite generale (exemple 8.9.2) : une boucle de rayon
`a`, dans le plan `Oxy`, parcourue par `I` dans le sens anti-horlogique.
Sur l'axe,

    B_z = mu_0 I a^2 / [2 (z^2 + a^2)^(3/2)]                       (8.9.14)

maximale au centre, ou elle vaut `mu_0 I / (2 a)`. Hors de l'axe, avec
`F = rho^2 + z^2 + a^2` et `K = 2 a rho / F` (8.9.12, 8.9.13),

    B_rho = (mu_0 I a z / 2 pi F^(3/2)) integrale de 0 a pi de
            cos(p) dp / (1 - K cos p)^(3/2)                        (8.9.9)
    B_z   = (mu_0 I a / 2 pi F^(3/2)) integrale de 0 a pi de
            (a - rho cos p) dp / (1 - K cos p)^(3/2)               (8.9.11)

« Ces integrales n'ont pas de solution analytique, il faut donc les
integrer numeriquement. »
"""

from __future__ import annotations

import math

import mpmath
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)


def induction_sur_l_axe(courant: object, rayon: object, cote: object) -> sp.Expr:
    """`B_z = mu_0 I a^2 / [2 (z^2 + a^2)^(3/2)]`."""
    a, z = sp.sympify(rayon), sp.sympify(cote)
    return (
        PERMEABILITE
        * sp.sympify(courant)
        * a**2
        / (2 * (z**2 + a**2) ** sp.Rational(3, 2))
    )


def induction_de_la_boucle(
    courant: float, rayon: float, rho: float, cote: float
) -> tuple[float, float]:
    """`(B_rho, B_z)` en teslas, par les integrales (8.9.9) et (8.9.11)."""
    somme = rho**2 + cote**2 + rayon**2
    module = 2 * rayon * rho / somme
    if module >= 1:
        msg = f"point sur le conducteur : rho = {rho}, z = {cote}"
        raise ValueError(msg)

    def poids(angle: float) -> mpmath.mpf:
        return mpmath.mpf(1 - module * mpmath.cos(angle)) ** -1.5

    facteur = PERMEABILITE_DU_VIDE * courant * rayon / (2 * math.pi * somme**1.5)
    bornes = [0, 0.05, 0.5, math.pi]
    radiale = mpmath.quad(lambda p: mpmath.cos(p) * poids(p), bornes)
    axiale = mpmath.quad(lambda p: (rayon - rho * mpmath.cos(p)) * poids(p), bornes)
    return (float(facteur * cote * radiale), float(facteur * axiale))
