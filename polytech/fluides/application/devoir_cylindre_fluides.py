"""Devoir cote de fluides : les fluides, le Reynolds, les regimes, le Mach.

Le devoir ne chiffre aucune donnee. Chacune vient d'un document du depot,
et `tests/test_sources_devoir_cylindre.py` retrouve chaque passage cite :

- l'air a vingt degres et une atmosphere, `mu = 1,83e-5 kg/(m s)`
  (`MecaniqueFluides/SLIDES/Slides_01_Introduction.pdf`, « Air at 20°C and
  1 atm ») ; sa masse volumique par le gaz parfait, `rho = p M / (R T)` ;
- l'air a une autre temperature : la correlation du TP 4 de thermique
  (`proprietes_tp_04.viscosite_de_l_air`), la meme masse volumique ;
- l'eau, `rho = 1000 kg/m3` (activite d'integration 1) et sa viscosite par
  la correlation du TP 4 ;
- la celerite du son, `c = racine(gamma R T / M)`, avec le `gamma = 1,4`
  du cours (`regimes`) ; l'eau n'en a pas ici, elle est incompressible.

Les regimes du cylindre isole viennent de deux planches qui se completent.
La planche 31 de la convection forcee
(`ThermiqueFondements/SLIDES/Slides_09_Convection_forcee.pdf`) : sous
`Re_d = 1e5`, couche limite laminaire et decollement vers 80 degres ; au
dela, transition avant decollement, vers 110 degres. La planche 32
d'aerodynamique (`Slides_08_Aerodynamique.pdf`) trace `C_D(Re)` : sa chute,
la crise de trainee, s'acheve vers `Re = 5e5` (point E, lu sur la courbe).
Sous `Re = 1`, l'ecoulement est rampant (`trainee.LIMITE_DE_STOKES`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from polytech.fluides.application import activite_2_hypotheses as hyp
from polytech.fluides.application.enonce import EAU as EAU_DE_L_ACTIVITE_1
from polytech.fluides.domain.grandeurs import Pression
from polytech.fluides.domain.proprietes import viscosite_cinematique
from polytech.fluides.domain.regime_de_reynolds import nombre_de_reynolds
from polytech.fluides.domain.regimes import GAMMA_DE_L_AIR, nombre_de_mach
from polytech.fluides.domain.statique.gaz import AIR as GAZ_AIR
from polytech.fluides.domain.statique.gaz import CONSTANTE_DES_GAZ
from polytech.fluides.domain.trainee import LIMITE_DE_STOKES
from polytech.thermique.application.proprietes_tp_04 import (
    viscosite_de_l_air,
    viscosite_de_l_eau,
)
from polytech.thermique.domain.grandeurs import ZERO_ABSOLU

ZERO_CELSIUS: Final = ZERO_ABSOLU
"""Le zero Celsius, ecrit une seule fois en thermique (`grandeurs`)."""
REYNOLDS_DE_TRANSITION: Final = 1e5
"""Planche 31 de thermique : la couche limite transite avant de decoller."""
REYNOLDS_DE_CRISE: Final = 5e5
"""Planche 32 d'aerodynamique, point E : la fin de la chute de `C_D`, lue."""


@dataclass(frozen=True, slots=True)
class Fluide:
    """Un fluide newtonien ; `celerite` vide s'il est traite incompressible."""

    nom: str
    masse_volumique: float
    viscosite: float
    celerite: float | None = None

    @property
    def viscosite_cinematique(self) -> float:
        """`nu = mu / rho`, en m^2/s."""
        return viscosite_cinematique(self.viscosite, self.masse_volumique)


def _air(nom: str, celsius: float, viscosite: float) -> Fluide:
    kelvins = celsius + ZERO_CELSIUS
    rho = GAZ_AIR.masse_volumique(Pression(pascals=hyp.AIR_PRESSION), kelvins)
    c = math.sqrt(GAMMA_DE_L_AIR * CONSTANTE_DES_GAZ / GAZ_AIR.masse_molaire * kelvins)
    return Fluide(nom, rho.kg_par_m3, viscosite, c)


def air_a(celsius: float) -> Fluide:
    """L'air a cette temperature, a une atmosphere (correlation du TP 4)."""
    return _air(f"air a {celsius:g} degres", celsius, viscosite_de_l_air(celsius))


AIR: Final = _air("air", hyp.AIR_TEMPERATURE - ZERO_CELSIUS, hyp.AIR_VISCOSITE)
EAU: Final = Fluide("eau", EAU_DE_L_ACTIVITE_1.kg_par_m3, viscosite_de_l_eau(20.0))


class Regime(Enum):
    """Les zones du cylindre isole, et l'angle de decollement qu'elles donnent."""

    RAMPANT = ("rampant", None)
    LAMINAIRE = ("couche limite laminaire", 80)
    TRANSITION = ("transition", 110)
    TURBULENT = ("couche limite turbulente", 110)

    @property
    def nom(self) -> str:
        """Le regime en toutes lettres."""
        return str(self.value[0])

    @property
    def decollement(self) -> int | None:
        """L'angle de decollement depuis l'amont, en degres (planche 31)."""
        angle = self.value[1]
        return None if angle is None else int(angle)


def reynolds(fluide: Fluide, vitesse: float, diametre: float) -> float:
    """`Re = U D / nu`, le diametre pour longueur caracteristique."""
    return nombre_de_reynolds(vitesse, diametre, fluide.viscosite_cinematique)


def regime(nombre: float) -> Regime:
    """La zone ou tombe ce Reynolds ; leve s'il est negatif."""
    if nombre < 0:
        msg = f"nombre de Reynolds negatif : {nombre}"
        raise ValueError(msg)
    seuils = (
        (LIMITE_DE_STOKES, Regime.RAMPANT),
        (REYNOLDS_DE_TRANSITION, Regime.LAMINAIRE),
        (REYNOLDS_DE_CRISE, Regime.TRANSITION),
    )
    return next((r for seuil, r in seuils if nombre < seuil), Regime.TURBULENT)


def mach(fluide: Fluide, vitesse: float) -> float | None:
    """`Ma = U / c` ; sans objet pour un fluide traite incompressible."""
    return None if fluide.celerite is None else nombre_de_mach(vitesse, fluide.celerite)
