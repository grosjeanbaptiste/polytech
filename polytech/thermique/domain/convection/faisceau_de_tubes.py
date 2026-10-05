"""La convection forcee a travers un faisceau de tubes.

Syllabus de thermique, tome 2, premiere partie, section 3.5. Des fumees
traversent un faisceau, perpendiculairement aux tubes, ranges en files
ou en quinconce. Les resultats de Grimison se mettent sous la forme

    Nu_d = C f_a f_n Re^m Pr^0,31,

avec `C = 0,34` et `m = 0,60` en files, `0,35` et `0,57` en quinconce.
Le Reynolds se forme sur la vitesse dans la section la plus retrecie,
entre deux tubes d'une meme rangee.

`f_a` est le facteur d'interaction, fonction des pas reduits
`a_T = p_T / d` et `a_L = p_L / d`. `f_n` est l'effet d'etablissement :
la turbulence du sillage croit de rangee en rangee, jusqu'a la dixieme.
Des ailettes longitudinales, mieux profilees, reduisent le sillage :
le coefficient en est multiplie par 0,83 environ.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Final


class Disposition(Enum):
    """Les deux mailles de la figure 3.9."""

    EN_FILES = "en files"
    EN_QUINCONCE = "en quinconce"


_ETABLISSEMENT: Final = {
    Disposition.EN_FILES: (0.64, 0.80, 0.87, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 1),
    Disposition.EN_QUINCONCE: (0.68, 0.75, 0.83, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99, 1),
}
"""Page 43 : `f_n` de la premiere a la dixieme rangee."""
_GRIMISON: Final = {
    Disposition.EN_FILES: (0.34, 0.60),
    Disposition.EN_QUINCONCE: (0.35, 0.57),
}
FACTEUR_DES_AILETTES_LONGITUDINALES: Final = 0.83
"""Page 44 : `f_h`, inferieur a l'unite."""


def vitesse_maximale(amont: float, *, pas: float, diametre: float) -> float:
    """`u_max = u_o p_T / (p_T - d)`."""
    if pas <= diametre:
        msg = f"tubes jointifs : pas {pas} m, diametre {diametre} m"
        raise ValueError(msg)
    return amont * pas / (pas - diametre)


def facteur_d_etablissement(rangees: int, disposition: Disposition) -> float:
    """`f_n` ; un au-dela de la dixieme rangee."""
    if rangees <= 0:
        msg = f"nombre de rangees nul ou negatif : {rangees}"
        raise ValueError(msg)
    table = _ETABLISSEMENT[disposition]
    return float(table[min(rangees, len(table)) - 1])


def facteur_d_interaction(
    disposition: Disposition,
    transversal: float,
    longitudinal: float,
    *,
    reynolds: float | None = None,
) -> float:
    """`f_a`, pour les pas reduits `a_T` et `a_L`."""
    if disposition is Disposition.EN_QUINCONCE:
        return 1 + 0.1 * transversal + 0.34 / longitudinal
    if reynolds is None:
        msg = "Reynolds requis pour des tubes en files"
        raise ValueError(msg)
    maille = transversal + 7.17 / transversal - 6.52
    serrage = 0.266 / (longitudinal - 0.8) ** 2 - 0.12
    return 1 + maille * serrage * math.sqrt(1000 / reynolds)


def nusselt_du_faisceau(
    disposition: Disposition, reynolds: float, prandtl: float, facteurs: float
) -> float:
    """`C f_a f_n Re^m Pr^0,31`, `facteurs` etant le produit `f_a f_n`."""
    coefficient, exposant = _GRIMISON[disposition]
    return float(coefficient * facteurs * reynolds**exposant * prandtl**0.31)
