"""Exercice cote 2019, question 1 : deux fils superposes devant un plan conducteur.

Le plan au potentiel zero est remplace par les images des deux fils. Le
potentiel d'un conducteur est alors la somme de quatre logarithmes, et
les coefficients de potentiel s'y lisent.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import (
    PERMITTIVITE,
    champ_par_gauss,
)

EPS: Final = PERMITTIVITE
H1, H2, RAYON = sp.symbols("h_1 h_2 R", positive=True)
Q1, Q2 = sp.symbols("lambda_1 lambda_2", real=True)
"""Les charges lineiques `Q1/L` et `Q2/L`."""
PERMITTIVITE_DU_MANUSCRIT: Final = 8.85e-12
"""`eps0`, en F/m, tel que le manuscrit l'ecrit : trois chiffres."""
HAUTEUR_DU_PREMIER: Final = 5.0
COEFFICIENT_VOULU: Final = 53e9
"""`K12 L`, en (F/m)^-1."""
CHARGE_LINEIQUE: Final = 3e-6


def _potentiel(charge: sp.Expr, vers_le_fil: sp.Expr, vers_l_image: sp.Expr) -> sp.Expr:
    """Le potentiel d'un fil et de son image, nul sur le plan."""
    return charge / (2 * sp.pi * EPS) * sp.log(vers_l_image / vers_le_fil)


def potentiel_du_premier() -> sp.Expr:
    """`V(1)` : `P` pose a la surface du conducteur 1, `R` petit devant les hauteurs."""
    propre = _potentiel(Q1, RAYON, 2 * H1)
    mutuel = _potentiel(Q2, H1 - H2, H1 + H2)
    return propre + mutuel


def coefficient_propre() -> sp.Expr:
    """`K11 L`, ce qui multiplie `Q1/L` dans `V(1)`."""
    return sp.simplify(potentiel_du_premier().diff(Q1))


def coefficient_mutuel() -> sp.Expr:
    """`K12 L`, ce qui multiplie `Q2/L` dans `V(1)`."""
    return sp.simplify(potentiel_du_premier().diff(Q2))


def hauteur() -> float:
    """`h2`, en metres, pour le `K12 L` voulu : `tanh` de l'exposant divise par deux."""
    exposant = 2 * math.pi * PERMITTIVITE_DU_MANUSCRIT * COEFFICIENT_VOULU
    return HAUTEUR_DU_PREMIER * math.expm1(exposant) / (math.exp(exposant) + 1)


def champ_en_p(hauteur: float) -> tuple[float, float]:
    """`(E_x, E_y)` en `P`, au pied des deux fils : `+Q/L` en haut, `-Q/L` dessous."""
    vertical = sp.Integer(0)
    for charge, distance in (
        (CHARGE_LINEIQUE, HAUTEUR_DU_PREMIER),
        (-CHARGE_LINEIQUE, hauteur),
    ):
        du_fil = -champ_par_gauss(charge, 1, distance)
        de_l_image = -champ_par_gauss(charge, 1, distance)
        vertical += du_fil + de_l_image
    return 0.0, float(vertical.subs(EPS, PERMITTIVITE_DU_MANUSCRIT))
