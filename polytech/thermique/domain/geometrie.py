"""La geometrie que les resistances traversent.

Un rayon n'est pas une epaisseur : l'un situe, l'autre mesure un
franchissement. Les distinguer n'est pas une coquetterie -- l'enonce du
TP 1 donne le **diametre** interieur et l'**epaisseur** de silice, et c'est
leur composition qui livre le rayon exterieur. Une addition faite de
travers y passerait inapercue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.thermique.domain.validation import exiger_strictement_positif

if TYPE_CHECKING:
    from polytech.thermique.domain.materiaux import Epaisseur


@dataclass(frozen=True, slots=True)
class Rayon:
    """Rayon d'une surface a symetrie spherique, en metres."""

    metres: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_strictement_positif(self.metres, "rayon")

    @classmethod
    def depuis_le_diametre(cls, metres: float) -> Rayon:
        """L'enonce donne un diametre : la division se fait une seule fois."""
        return cls(metres=metres / 2.0)

    @property
    def surface_de_sphere(self) -> float:
        """L'aire 4 pi r^2 de la sphere de ce rayon, en m^2."""
        return 4.0 * math.pi * self.metres**2

    def augmente_de(self, epaisseur: Epaisseur) -> Rayon:
        """Le rayon atteint apres avoir traverse cette epaisseur."""
        return Rayon(metres=self.metres + epaisseur.metres)
