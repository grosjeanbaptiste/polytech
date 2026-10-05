"""Le signe d'une inductance mutuelle, lu sur les bornes pointees.

Les planches du chapitre III.4 (16 a 20). Pour representer deux circuits
couples, on marque d'un point la borne d'entree de chacun : si le courant
entre dans l'un par sa borne pointee, il fait passer a travers l'autre un
flux dans le sens positif de celui-ci.

L'inductance mutuelle est donc **positive** quand les deux courants
entrent par les points, ou en sortent tous deux, et negative sinon. Les
equations electromagnetiques s'ecrivent `phi_1 = L_1 I_1 +- L_12 I_2`, et
deux inductances couplees en serie valent `L_1 + L_2 +- 2 L_12`.
"""

from __future__ import annotations

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def signe_de_la_mutuelle(
    *, premier_par_le_point: bool, second_par_le_point: bool
) -> int:
    """`+1` si les deux courants entrent, ou sortent, par les bornes pointees."""
    return 1 if premier_par_le_point == second_par_le_point else -1


def flux_des_circuits_couples(
    propres: tuple[Grandeur, Grandeur],
    mutuelle: Grandeur,
    courants: tuple[Grandeur, Grandeur],
    *,
    premier_par_le_point: bool,
    second_par_le_point: bool,
) -> tuple[Grandeur, Grandeur]:
    """`(phi_1, phi_2)`, la mutuelle prise positive et son signe lu sur les points."""
    signe = signe_de_la_mutuelle(
        premier_par_le_point=premier_par_le_point,
        second_par_le_point=second_par_le_point,
    )
    premier = propres[0] * courants[0] + signe * mutuelle * courants[1]
    second = signe * mutuelle * courants[0] + propres[1] * courants[1]
    return premier, second


def inductance_en_serie(
    propres: tuple[Grandeur, Grandeur],
    mutuelle: Grandeur,
    *,
    premier_par_le_point: bool,
    second_par_le_point: bool,
) -> Grandeur:
    """`L_1 + L_2 +- 2 L_12` : le flux total d'un meme courant, rapporte a lui."""
    flux = flux_des_circuits_couples(
        propres,
        mutuelle,
        (1, 1),
        premier_par_le_point=premier_par_le_point,
        second_par_le_point=second_par_le_point,
    )
    return flux[0] + flux[1]
