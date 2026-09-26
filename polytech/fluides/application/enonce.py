"""Donnees numeriques de l'activite d'integration 1.

Un seul endroit ou l'enonce est transcrit : toute correction se fait ici et
se propage au document par la generation des macros.
"""

from __future__ import annotations

from typing import Final

from polytech.fluides.domain.cinematique.champ import ChampDeVitesse
from polytech.fluides.domain.cinematique.symboles import T, X, Y
from polytech.fluides.domain.grandeurs import Hauteur, MasseVolumique, Pression
from polytech.fluides.domain.statique.manometre import Manometre, Sens, Troncon

EAU: Final = MasseVolumique(kg_par_m3=1_000.0)
HUILE: Final = MasseVolumique(kg_par_m3=850.0)
MERCURE: Final = MasseVolumique(kg_par_m3=13_600.0)

PRESSION_ATMOSPHERIQUE: Final = Pression(pascals=85_600.0)
"""Pression atmospherique a 1400 m d'altitude, donnee de l'enonce."""

PRESSION_NIVEAU_MER: Final = Pression(pascals=101_300.0)
"""Reference au niveau de la mer, pour la discussion sur l'altitude.

Atmosphere type (ISO 2533 / OACI) : 101 325 Pa, arrondi a 101,3 kPa.
"""

REPONSE_DU_COURS: Final = Pression(pascals=130_000.0)
"""Reponse publiee par le cours pour cet exercice.

Planche II.3 « Instruments de mesure de la pression : le manometre
multi-fluides », mention « Rep. : p1 = 130 kPa ». Elle sert de
corroboration independante du resultat que la bibliotheque calcule.
"""

MANOMETRE: Final = Manometre(
    troncons=(
        Troncon(EAU, Hauteur(metres=0.10), Sens.DESCENTE, nom="eau"),
        Troncon(HUILE, Hauteur(metres=0.20), Sens.DESCENTE, nom="huile"),
        Troncon(MERCURE, Hauteur(metres=0.35), Sens.MONTEE, nom="Hg"),
    ),
)

CHAMP_B: Final = ChampDeVitesse(u=X * T, v=X - Y)
CHAMP_C: Final = ChampDeVitesse(u=2 * X, v=-2 * Y)

POINT_B: Final = (1.0, 2.0, 2.0)
"""Point (x, y) et instant t ou l'enonce demande l'acceleration."""
