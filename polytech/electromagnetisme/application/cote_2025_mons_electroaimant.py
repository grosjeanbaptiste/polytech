"""Exercice cote 2025 (Mons), question 2 : un electro-aimant et sa plaque.

Un demi-anneau d'hipersil, deux entrefers de 2 mm, une plaque de toles
ordinaires. Meme flux et meme section partout : l'induction est la meme
dans l'hipersil, l'air et la plaque ; seul le champ differe.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    potentiel_scalaire_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.entrefer import (
    champ_dans_l_entrefer,
)
from polytech.fluides.domain.grandeurs import PESANTEUR

CHAMP_DANS_L_HIPERSIL: Final = 4.0
"""`H_H`, en A/m, donne par l'enonce."""
INDUCTION_LUE: Final = 0.3
"""`B_H`, en teslas, **lue sur la figure 2.3** (hipersil) pour 4 A/m."""
CHAMP_LU_DANS_LES_TOLES: Final = 55.0
"""`H_TO`, en A/m, **lu sur la figure 2.2** (toles ordinaires) pour 0,3 T."""
SECTION: Final = 2e-2 * 2e-2
RAYON_MOYEN: Final = 7e-2
LONGUEUR_DANS_LA_PLAQUE: Final = (1 + 14 + 1) * 1e-2
"""D'un pole a l'autre a mi-epaisseur : 1 cm, 14 cm, 1 cm."""
ENTREFER: Final = 2e-3
SPIRES: Final = 200
MASSE: Final = 0.5


def flux() -> float:
    """`Phi = B_H S`, en webers ; sans dispersion, le meme partout."""
    return INDUCTION_LUE * SECTION


def induction_dans_la_plaque() -> float:
    """`B_TO = Phi / S_TO`, la plaque ayant la section de l'anneau."""
    return flux() / SECTION


def partage() -> tuple[float, float, float]:
    """Les amperes-tours de l'hipersil, des toles, puis des deux entrefers."""
    hipersil = potentiel_scalaire_magnetique(
        CHAMP_DANS_L_HIPERSIL, math.pi * RAYON_MOYEN
    )
    toles = potentiel_scalaire_magnetique(
        CHAMP_LU_DANS_LES_TOLES, LONGUEUR_DANS_LA_PLAQUE
    )
    air = potentiel_scalaire_magnetique(
        champ_dans_l_entrefer(induction_dans_la_plaque()), 2 * ENTREFER
    )
    return hipersil, toles, air


def courant() -> float:
    """Le theoreme d'Ampere : la somme des amperes-tours vaut `N I`."""
    return sum(partage()) / SPIRES


def force(intensite: float) -> float:
    """`|F| = (1/4) I^2 N^2 mu0 S / le^2`, l'air portant tous les amperes-tours."""
    return (
        (intensite * SPIRES) ** 2 * PERMEABILITE_DU_VIDE * SECTION / (4 * ENTREFER**2)
    )


def induction_approchee(intensite: float) -> float:
    """`B = mu0 N I / (2 le)`, en teslas : l'air porte tous les amperes-tours."""
    return PERMEABILITE_DU_VIDE * SPIRES * intensite / (2 * ENTREFER)


def force_par_la_pression(intensite: float) -> float:
    """`B^2 / (2 mu0)` sur chacun des deux poles."""
    pression = induction_approchee(intensite) ** 2 / (2 * PERMEABILITE_DU_VIDE)
    return 2 * pression * SECTION


def poids() -> float:
    """`M g`, en newtons."""
    return MASSE * PESANTEUR


def courant_minimal() -> float:
    """Le courant dont la force egale le poids : la force va comme `I^2`."""
    return math.sqrt(poids() / force(1.0))
