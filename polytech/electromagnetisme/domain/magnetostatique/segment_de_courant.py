"""L'induction et le potentiel vecteur d'un segment de droite parcouru par un courant.

Le syllabus d'electricite generale (exemple 8.9.1) : un segment porte par
l'axe `z`, de `z = -l_1` a `z = +l_2`, parcouru par un courant `I`. En
coordonnees cylindriques, `R_1` et `R_2` etant les distances du point aux
deux extremites :

    B_phi = (mu_0 I / 4 pi rho) [(l_2 - z) / R_2 + (l_1 + z) / R_1]      (8.9.4)
    A_z   = (mu_0 I / 4 pi) ln[(l_2 - z + R_2) / (-(l_1 + z) + R_1)]     (8.9.6)

Pour une droite illimitee, `B_phi = mu_0 I / (2 pi rho)` (8.9.7) ; le
potentiel vecteur diverge, et se reduit a `(mu_0 I / 2 pi) ln(rho_0 / rho)`
(8.9.8). Un segment seul n'est pas un circuit : il ne vaut que comme
morceau d'un circuit ferme, par exemple un cote de spire carree.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.symboles import COTE, RAYON


def _distances(bas: object, haut: object) -> tuple[sp.Expr, sp.Expr]:
    vers_le_bas = sp.sqrt(RAYON**2 + (COTE + sp.sympify(bas)) ** 2)
    vers_le_haut = sp.sqrt(RAYON**2 + (COTE - sp.sympify(haut)) ** 2)
    return (vers_le_bas, vers_le_haut)


def induction_du_segment(courant: object, bas: object, haut: object) -> sp.Expr:
    """`B_phi(rho, z)` -- relation (8.9.4)."""
    r_1, r_2 = _distances(bas, haut)
    facteur = PERMEABILITE * sp.sympify(courant) / (4 * sp.pi * RAYON)
    return facteur * ((haut - COTE) / r_2 + (bas + COTE) / r_1)


def potentiel_vecteur_du_segment(courant: object, bas: object, haut: object) -> sp.Expr:
    """`A_z(rho, z)` -- relation (8.9.6)."""
    r_1, r_2 = _distances(bas, haut)
    quotient = (haut - COTE + r_2) / (-(bas + COTE) + r_1)
    return PERMEABILITE * sp.sympify(courant) / (4 * sp.pi) * sp.log(quotient)


def induction_de_la_droite(courant: object) -> sp.Expr:
    """`B_phi = mu_0 I / (2 pi rho)`, pour une droite illimitee (8.9.7)."""
    return PERMEABILITE * sp.sympify(courant) / (2 * sp.pi * RAYON)


def potentiel_vecteur_de_la_droite(courant: object, reference: object) -> sp.Expr:
    """`A_z = (mu_0 I / 2 pi) ln(rho_0 / rho)`, nul a la distance `rho_0` (8.9.8)."""
    facteur = PERMEABILITE * sp.sympify(courant) / (2 * sp.pi)
    return facteur * sp.log(sp.sympify(reference) / RAYON)


def induction_au_centre_d_une_spire_carree(courant: object, cote: object) -> sp.Expr:
    """`B = 2 sqrt(2) mu_0 I / (pi l)` : quatre cotes, chacun vu de `l / 2`."""
    return (
        2 * sp.sqrt(2) * PERMEABILITE * sp.sympify(courant) / (sp.pi * sp.sympify(cote))
    )
