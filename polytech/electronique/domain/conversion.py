"""Le pas de quantification d'un convertisseur analogique-numerique.

Le corrige ecrit le pas exact -- pleine echelle sur `2^n - 1` -- puis
l'approxime par `2^n`. Les deux sont ici, parce que l'approximation est une
affirmation comme une autre : elle doit pouvoir se chiffrer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConvertisseurAnalogiqueNumerique:
    """Un convertisseur, defini par sa resolution et sa pleine echelle."""

    bits: int
    pleine_echelle_volts: float

    def __post_init__(self) -> None:
        """Valide les invariants de l'objet a sa construction."""
        if self.bits <= 0:
            msg = f"resolution non strictement positive : {self.bits} bits"
            raise ValueError(msg)
        if self.pleine_echelle_volts <= 0:
            msg = f"pleine echelle non positive : {self.pleine_echelle_volts} V"
            raise ValueError(msg)

    def niveaux(self) -> int:
        """Nombre de niveaux representables : deux puissance le nombre de bits."""
        return int(2**self.bits)

    def pas_exact(self) -> float:
        """Pas de quantification exact : la pleine echelle sur `2^n - 1`.

        Les `2^n` niveaux delimitent `2^n - 1` intervalles ; c'est entre les
        bornes extremes que la pleine echelle se repartit.
        """
        return self.pleine_echelle_volts / (self.niveaux() - 1)

    def pas_approche(self) -> float:
        """Le meme pas, approxime par `2^n`, comme le fait le corrige."""
        return self.pleine_echelle_volts / self.niveaux()
