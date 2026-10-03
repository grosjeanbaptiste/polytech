"""Les donnees du TP 4 : les fluides, et les quatre batteries proposees.

Source : `ThermiqueFondements/TP/TP_04_Echangeur_Enonce.pdf`, pages 4 et
5 (des images, lues sur le PDF). Les correlations des fluides prennent
la temperature en degres Celsius, comme l'enonce le souligne en rouge.

La hauteur de la gaine n'est pas dans l'enonce : l'officiel la prend
egale au nombre de tubes d'un rang fois le cote des ailettes, et ce
document fait de meme pour les quatre batteries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


def chaleur_de_l_air(t: float) -> float:
    """`c = 1006,6 + 8e-4 (t - 10)^2`, J/(kg K)."""
    return 1006.6 + 8e-4 * (t - 10) ** 2


def conductivite_de_l_air(t: float) -> float:
    """`lambda = 0,0245 + 0,75e-4 t`, W/(m K)."""
    return 0.0245 + 0.75e-4 * t


def viscosite_de_l_air(t: float) -> float:
    """`mu = 1,71e-5 + 0,45e-7 t`, Pa s."""
    return 1.71e-5 + 0.45e-7 * t


def chaleur_de_l_eau(t: float) -> float:
    """`c = 4176 + 6,16e-3 (t - 10)^2`, J/(kg K)."""
    return 4176 + 6.16e-3 * (t - 10) ** 2


def conductivite_de_l_eau(t: float) -> float:
    """`lambda = 0,585 + 9,38e-3 racine(t - 10)` ; leve sous dix degres."""
    if t < 10:  # noqa: PLR2004 -- la borne de la racine, dans la formule meme
        msg = f"correlation de l'eau hors de son domaine : {t} < 10 degres"
        raise ValueError(msg)
    return 0.585 + 9.38e-3 * math.sqrt(t - 10)


def viscosite_de_l_eau(t: float) -> float:
    """`mu = 27,531e-6 exp(527,93 / (126,53 + t))`, Pa s."""
    return 27.531e-6 * math.exp(527.93 / (126.53 + t))


@dataclass(frozen=True, slots=True)
class Batterie:
    """Un rang de tubes ailettes, par metre de tube pour les surfaces."""

    cote: float
    pas: float
    epaisseur: float
    tubes_par_rang: int
    interieure: float
    primaire: float
    ailettes: float
    nusselt: tuple[float, float, float, float, float]
    """Le polynome de la page 5, du degre quatre au degre zero."""


DIAMETRE_INTERIEUR, DIAMETRE_EXTERIEUR = 0.014, 0.016
CONDUCTIVITE_TUBE, CONDUCTIVITE_AILETTE, LONGUEUR_DE_TUBE = 380.0, 210.0, 0.5
BATTERIES: Final = {
    "TCPL16-40-4-50": Batterie(
        0.040,
        0.005,
        0.0004,
        7,
        4.398e-2,
        4.624e-2,
        5.596e-1,
        (1.3489e-12, -4.7666e-09, 4.7237e-06, 5.4432e-03, 1.7189),
    ),
    "TCPL16-45-2-25": Batterie(
        0.045,
        0.0025,
        0.0002,
        6,
        4.398e-2,
        4.624e-2,
        1.459,
        (-6.0606e-12, 1.3788e-08, -1.0439e-05, 6.9002e-03, 2.0890),
    ),
    "TCPL16-40-4-25": Batterie(
        0.040,
        0.0025,
        0.0004,
        7,
        4.398e-2,
        4.222e-2,
        1.119,
        (-8.4645e-12, 1.8441e-08, -1.2167e-05, 6.5314e-03, 1.9008),
    ),
    "TCPL16-45-4-25": Batterie(
        0.045,
        0.0025,
        0.0004,
        6,
        4.398e-2,
        4.222e-2,
        1.459,
        (-1.3782e-11, 3.2574e-08, -2.5309e-05, 1.0911e-02, 1.4633),
    ),
}
