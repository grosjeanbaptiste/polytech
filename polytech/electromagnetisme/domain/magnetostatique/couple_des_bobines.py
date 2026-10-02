"""Le couple entre deux bobines couplees -- seance 2.1, exercice 17.

L'energie de deux circuits couples vaut
`W = (L_11 I_1^2 + L_22 I_2^2 + 2 L_12 I_1 I_2) / 2`. Quand l'une des
bobines tourne, seule la mutuelle depend de l'angle : les inductances
propres disparaissent a la derivation, et le couple a courants constants
se reduit a `I_1 I_2 dL_12/d(alpha)`.

**L'unite de la pente est le piege.** Un couple est une energie par
radian. Une pente donnee par degre doit etre convertie, sans quoi le
resultat est `180 / pi` fois trop petit -- un facteur cinquante-sept que
rien ne signale, le resultat restant un nombre plausible.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp


def pente_par_radian(pente_par_degre: float) -> float:
    """Convertit une pente par degre en pente par radian.

    Un radian vaut `180 / pi` degres : sur un radian, la grandeur varie
    donc `180 / pi` fois plus que sur un degre.
    """
    return pente_par_degre * 180 / math.pi


def energie_de_deux_bobines(
    propre_1: sp.Expr,
    propre_2: sp.Expr,
    mutuelle: sp.Expr,
    courant_1: sp.Expr,
    courant_2: sp.Expr,
) -> sp.Expr:
    """`W = (L_11 I_1^2 + L_22 I_2^2 + 2 L_12 I_1 I_2) / 2`."""
    propres = propre_1 * courant_1**2 + propre_2 * courant_2**2
    return (propres + 2 * mutuelle * courant_1 * courant_2) / 2


def couple_a_courants_constants(
    courant_1: float,
    courant_2: float,
    pente_de_la_mutuelle: float,
) -> float:
    """`C = I_1 I_2 dL_12/d(alpha)`, la pente etant **par radian**.

    Le signe est positif devant la derivee : a courants constants, les
    sources fournissent du travail en plus de l'energie stockee.
    """
    return courant_1 * courant_2 * pente_de_la_mutuelle
