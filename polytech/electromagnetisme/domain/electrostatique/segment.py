"""Le champ et le potentiel d'un segment de droite uniformement charge.

Le syllabus d'electricite generale (exemple 2.6.2) : un segment porte par
l'axe `z`, de `z = -l_1` a `z = +l_2`, de charge lineique `lambda`. En
coordonnees cylindriques `(rho, z)`, avec `R_1` et `R_2` les distances du
point aux deux extremites :

    E_rho = (lambda / 4 pi eps_0 rho) [(l_2 - z) / R_2 + (l_1 + z) / R_1],
    E_z   = (lambda / 4 pi eps_0) [1 / R_2 - 1 / R_1],
    V     = (lambda / 4 pi eps_0) ln[(z + l_1 + R_1) / (z - l_2 + R_2)].

Quand le segment devient une droite, le champ tend vers
`lambda / (2 pi eps_0 rho)` (2.6.12) ; le potentiel, lui, diverge, et seule
la difference `U_12 = (lambda / 2 pi eps_0) ln(rho_2 / rho_1)` garde un
sens (2.6.19) : c'est le potentiel logarithmique.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB
from polytech.electromagnetisme.domain.symboles import COTE, RAYON


def _distances(bas: object, haut: object) -> tuple[sp.Expr, sp.Expr]:
    vers_le_bas = sp.sqrt(RAYON**2 + (COTE + sp.sympify(bas)) ** 2)
    vers_le_haut = sp.sqrt(RAYON**2 + (COTE - sp.sympify(haut)) ** 2)
    return (vers_le_bas, vers_le_haut)


def champ_du_segment(
    densite: object, bas: object, haut: object
) -> tuple[sp.Expr, sp.Expr]:
    """`(E_rho, E_z)` au point `(rho, z)` -- relations (2.6.10) et (2.6.11)."""
    r_1, r_2 = _distances(bas, haut)
    facteur = COULOMB * sp.sympify(densite)
    radial = facteur / RAYON * ((haut - COTE) / r_2 + (bas + COTE) / r_1)
    axial = facteur * (1 / r_2 - 1 / r_1)
    return (radial, axial)


def potentiel_du_segment(densite: object, bas: object, haut: object) -> sp.Expr:
    """`V(rho, z)`, nul a l'infini -- relation (2.6.13)."""
    r_1, r_2 = _distances(bas, haut)
    quotient = (COTE + bas + r_1) / (COTE - haut + r_2)
    return COULOMB * sp.sympify(densite) * sp.log(quotient)


def potentiel_par_les_distances(densite: object, bas: object, haut: object) -> sp.Expr:
    """`V = (lambda / 4 pi eps_0) ln[(R_1 + R_2 + l) / (R_1 + R_2 - l)]` (2.6.15)."""
    r_1, r_2 = _distances(bas, haut)
    longueur = sp.sympify(bas) + sp.sympify(haut)
    quotient = (r_1 + r_2 + longueur) / (r_1 + r_2 - longueur)
    return COULOMB * sp.sympify(densite) * sp.log(quotient)


def champ_de_la_droite_chargee(densite: object) -> sp.Expr:
    """`E_rho = lambda / (2 pi eps_0 rho)`, pour une droite illimitee (2.6.12)."""
    return sp.sympify(densite) / (2 * sp.pi * PERMITTIVITE * RAYON)


def tension_entre_deux_rayons(
    densite: object, premier: object, second: object
) -> sp.Expr:
    """`U_12 = V_1 - V_2 = (lambda / 2 pi eps_0) ln(rho_2 / rho_1)` (2.6.19)."""
    rho_1, rho_2 = sp.sympify(premier), sp.sympify(second)
    for rayon in (rho_1, rho_2):
        if rayon.is_positive is False:
            msg = f"rayon strictement positif attendu, et non {rayon}"
            raise ValueError(msg)
    return sp.sympify(densite) / (2 * sp.pi * PERMITTIVITE) * sp.log(rho_2 / rho_1)
