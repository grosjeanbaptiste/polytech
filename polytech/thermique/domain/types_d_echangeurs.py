"""La classification des echangeurs de chaleur, et leur compacite.

Planches du jeu 11 (planches 3 a 12). Un echangeur se classe d'abord par
la facon dont il met ses deux fluides en contact :

* **indirect** : une paroi les separe -- le cas le plus frequent, et le
  seul que le cours calcule ;
* **direct** : aucune paroi ; les fluides doivent pouvoir se meler et se
  separer ensuite, comme l'eau et l'air d'un refrigerant atmospherique ;
* **periodique** : une masse accumulatrice passe tour a tour dans le
  fluide chaud et dans le froid -- les regenerateurs des fours a
  1500 degres, ou aucun metal ne tiendrait.

Puis par la configuration de l'ecoulement, et par sa **compacite** :
l'aire d'echange logee dans un metre cube. Les fourchettes ci-dessous
sont celles des planches.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping


class Contact(Enum):
    """Le mode de mise en contact des deux fluides."""

    INDIRECT = "fluides separes par une paroi"
    DIRECT = "pas de paroi de separation"
    PERIODIQUE = "masse accumulatrice, en contact alternativement"

    @property
    def separe_les_fluides(self) -> bool:
        """Faux pour le seul contact direct, ou le melange doit etre permis."""
        return self is not Contact.DIRECT


class Ecoulement(Enum):
    """La configuration d'ecoulement des deux fluides."""

    COCOURANT = "paralleles, de meme sens"
    CONTRE_COURANT = "paralleles, de sens opposes"
    COURANTS_CROISES = "perpendiculaires"

    @property
    def dimensions_des_profils(self) -> int:
        """`t = f(x)` en courants paralleles, `t = f(x, y)` en courants croises."""
        return 2 if self is Ecoulement.COURANTS_CROISES else 1


class Realisation(Enum):
    """Les technologies dont les planches chiffrent la compacite."""

    TUBES_ET_VIROLE = "faisceau de tubes dans une virole cylindrique"
    PLAQUES = "empilage de plaques corruguees"
    PLAQUES_ET_AILETTES = "plaques ondulees, brasees"


COMPACITES: Final[Mapping[Realisation, tuple[float, float]]] = {
    Realisation.TUBES_ET_VIROLE: (10.0, 200.0),
    Realisation.PLAQUES: (200.0, 400.0),
    Realisation.PLAQUES_ET_AILETTES: (100.0, 1000.0),
}
"""En m^2 d'aire d'echange par m^3 d'appareil."""


def compacite(surface: float, volume: float) -> float:
    """L'aire d'echange par unite de volume, en m^2/m^3."""
    if volume <= 0:
        msg = f"volume nul ou negatif : {volume} m^3"
        raise ValueError(msg)
    return surface / volume


def realisations_de_compacite(valeur: float) -> tuple[Realisation, ...]:
    """Les technologies dont la fourchette contient cette compacite."""
    return tuple(
        realisation
        for realisation, (basse, haute) in COMPACITES.items()
        if basse <= valeur <= haute
    )
