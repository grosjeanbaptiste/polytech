"""Exercice cote 2021 (Mons), question 1 : une tige qui glisse sur deux rails.

Le champ sort de la feuille (`+z`), la tige est le long de `y` et glisse
vers la gauche (`-x`) : la spire qu'elle ferme avec le fil retrecit.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.induction import (
    fem_de_deplacement,
    force_electromotrice,
)
from polytech.electromagnetisme.domain.magnetostatique.lois import (
    force_sur_un_element_de_courant,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

ECARTEMENT: Final = 0.5
"""`delta`, la distance entre les rails, en metres."""
INDUCTION: Final = 1.5
RESISTANCE: Final = 10.0
VITESSE: Final = (-30.0, 0.0, 0.0)
"""La vitesse de la tige, en m/s : vers la gauche."""


def fem() -> float:
    """`e = B delta v` : le champ electromoteur integre le long de la tige."""
    return fem_de_deplacement(INDUCTION, ECARTEMENT, abs(VITESSE[0]))


def fem_par_le_flux() -> sp.Expr:
    """`-d(phi)/dt`, la spire ayant la longueur `l0 - v t`."""
    longueur = sp.Symbol("l_0", positive=True) + VITESSE[0] * TEMPS
    return force_electromotrice(INDUCTION * ECARTEMENT * longueur)


def courant() -> float:
    """`I = e / R`, la tige etant la seule resistance de la spire."""
    return fem() / RESISTANCE


def puissance() -> float:
    """`P = R I^2`, dissipee dans la tige."""
    return RESISTANCE * courant() ** 2


def force() -> tuple[float, float, float]:
    """`I l ^ B` sur la tige ; le courant y monte, selon `+y`.

    La force electromotrice est positive pour une normale selon `+z` : le
    courant tourne dans le sens trigonometrique, donc monte dans la tige,
    qui ferme la spire par la droite.
    """
    return force_sur_un_element_de_courant(
        courant(), (0.0, ECARTEMENT, 0.0), (0.0, 0.0, INDUCTION)
    )


def puissance_mecanique() -> float:
    """Ce qu'il faut fournir pour maintenir la vitesse contre la force."""
    return -sum(f * v for f, v in zip(force(), VITESSE, strict=True))
