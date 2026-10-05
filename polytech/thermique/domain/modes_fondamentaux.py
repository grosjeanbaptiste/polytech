"""Les deux modes fondamentaux du transfert de chaleur, et ce qu'on leur demande.

Planches d'introduction du cours de thermique. La thermodynamique dit
vers quel equilibre un systeme isole tend ; la thermique dit a quelle
vitesse. Le cours nomme **deux** modes fondamentaux :

* la conductibilite, transfert de proche en proche, qui exige un milieu
  materiel -- pure dans un milieu au repos, elle devient convection
  quand le fluide en mouvement emporte la chaleur avec lui ;
* le rayonnement, transfert a distance par onde electromagnetique, qui
  traverse le vide.

Une couche de fluide entre deux plaques horizontales reste au repos tant
que la plus chaude est au-dessus : le fluide leger y est deja. Chauffee
par le dessous, elle est instable et convecte.

Tout equipement qui degrade de l'energie en chaleur doit l'evacuer : la
part `1 - eta` de ce qu'une machine traite, ou, pour un corps a
l'effort, ce que son rendement mecanique ne convertit pas.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class ModeFondamental(Enum):
    """Les deux modes de la planche 19."""

    CONDUCTIBILITE = "de proche en proche, dans un milieu materiel"
    RAYONNEMENT = "a distance, par onde electromagnetique"

    @property
    def exige_un_milieu(self) -> bool:
        """Vrai pour la conductibilite : le rayonnement traverse le vide."""
        return self is ModeFondamental.CONDUCTIBILITE


class Transmission(Enum):
    """Les trois transmissions que le cours etudie, rapportees a leur mode."""

    CONDUCTIBILITE_PURE = "milieu au repos, indeformable"
    CONVECTION = "milieu deformable en mouvement"
    RAYONNEMENT = "entre surfaces opaques, a travers un milieu transparent"

    @property
    def mode(self) -> ModeFondamental:
        """La convection est de la conductibilite, renforcee par l'advection."""
        if self is Transmission.RAYONNEMENT:
            return ModeFondamental.RAYONNEMENT
        return ModeFondamental.CONDUCTIBILITE

    @property
    def milieu_en_mouvement(self) -> bool:
        """Vrai pour la seule convection."""
        return self is Transmission.CONVECTION


def temperature_d_equilibre(
    capacites: Sequence[float], temperatures: Sequence[float]
) -> float:
    """`t_m`, la moyenne des temperatures ponderee par les capacites `m c`."""
    for capacite in capacites:
        if capacite <= 0:
            msg = f"capacite nulle ou negative : {capacite} J/K"
            raise ValueError(msg)
    ponderee = sum(c * t for c, t in zip(capacites, temperatures, strict=True))
    return ponderee / sum(capacites)


def couche_au_repos(*, dessus: float, dessous: float) -> bool:
    """Vrai si la plaque du dessus n'est pas la plus froide : conduction pure."""
    return dessus >= dessous


def _exiger_un_rendement(rendement: float) -> None:
    if not 0 < rendement <= 1:
        msg = f"rendement hors de ]0, 1] : {rendement}"
        raise ValueError(msg)


def puissance_a_evacuer(puissance: float, rendement: float) -> float:
    """`(1 - eta) P` : ce qu'une machine degrade en chaleur."""
    _exiger_un_rendement(rendement)
    return (1 - rendement) * puissance


def chaleur_de_l_activite(travail: float, rendement: float) -> float:
    """`W (1 / eta - 1)` : la chaleur qui accompagne un travail mecanique `W`."""
    _exiger_un_rendement(rendement)
    return travail * (1 / rendement - 1)
