"""Objets-valeur de la thermique : immuables, egaux par leurs attributs.

Deux invariants portent une physique.

La **temperature absolue** ne descend pas sous le zero kelvin, et l'objet
ne connait qu'une echelle -- le kelvin. L'enonce du TP 1 donne ses
temperatures en degres Celsius pour le charbon et en kelvins pour
l'hydrogene, dans le meme document : une soustraction qui melange les deux
se trompe de 273,15 sans rien signaler.

Le **coefficient d'echange nul** est au contraire licite : c'est ce que
veut dire « paroi adiabatique ». Le refuser obligerait a traiter le cas
(a) du TP comme un autre probleme, alors qu'il est le meme avec K = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polytech.thermique.domain.validation import exiger_fini

ZERO_ABSOLU: Final = 273.15
"""Decalage entre l'echelle Celsius et l'echelle absolue, en kelvins."""


@dataclass(frozen=True, slots=True)
class EcartDeTemperature:
    """Difference de temperature, en kelvins : peut etre negative.

    Un ecart ne depend pas de l'echelle -- vingt degres d'ecart valent
    vingt kelvins. C'est ce qui rend la soustraction sure.
    """

    kelvins: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_fini(self.kelvins, "ecart de temperature")


@dataclass(frozen=True, slots=True)
class Temperature:
    """Temperature absolue, en kelvins."""

    kelvins: float

    def __post_init__(self) -> None:
        """Valide l'invariant a la construction."""
        exiger_fini(self.kelvins, "temperature")
        if self.kelvins < 0:
            msg = f"temperature absolue negative interdite : {self.kelvins} K"
            raise ValueError(msg)

    @classmethod
    def en_kelvins(cls, kelvins: float) -> Temperature:
        """Construit depuis l'echelle absolue."""
        return cls(kelvins=kelvins)

    @classmethod
    def en_celsius(cls, celsius: float) -> Temperature:
        """Construit depuis l'echelle Celsius, telle que l'enonce l'ecrit."""
        return cls(kelvins=celsius + ZERO_ABSOLU)

    @property
    def celsius(self) -> float:
        """Meme temperature en degres Celsius."""
        return self.kelvins - ZERO_ABSOLU

    def ecart_a(self, autre: Temperature) -> EcartDeTemperature:
        """L'ecart de cette temperature a une autre."""
        return EcartDeTemperature(kelvins=self.kelvins - autre.kelvins)
