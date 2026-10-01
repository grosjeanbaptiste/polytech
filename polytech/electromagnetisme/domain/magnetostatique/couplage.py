"""Le couplage magnetique : inductance mutuelle et coefficient de couplage.

`k = M / racine(L1 L2)` ne peut pas depasser l'unite -- un couplage
parfait transfere tout le flux, il n'en cree pas. La garde n'est donc pas
formelle : un `k > 1` signale une donnee mal lue ou une mutuelle calculee
de travers, et le laisser passer donnerait une inductance equivalente
negative.

Le signe de `M` dans la mise en serie depend du sens des enroulements :
`L1 + L2 + 2M` s'ils s'aident, `L1 + L2 - 2M` s'ils s'opposent. C'est le
point que les exercices verifient, et c'est pourquoi le sens est un
argument explicite et non une convention tacite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CouplageMagnetique:
    """Deux inductances et leur mutuelle."""

    premiere: float
    seconde: float
    mutuelle: float

    def __post_init__(self) -> None:
        """Refuse une inductance negative et un couplage superieur a un."""
        for valeur, nom in ((self.premiere, "premiere"), (self.seconde, "seconde")):
            if valeur <= 0:
                msg = f"inductance {nom} nulle ou negative : {valeur} H"
                raise ValueError(msg)
        if abs(self.mutuelle) > math.sqrt(self.premiere * self.seconde):
            msg = (
                f"couplage superieur a l'unite : M = {self.mutuelle} depasse "
                f"racine(L1 L2) = {math.sqrt(self.premiere * self.seconde):.6g}"
            )
            raise ValueError(msg)

    def coefficient(self) -> float:
        """`k = M / racine(L1 L2)`, entre zero et un."""
        return self.mutuelle / math.sqrt(self.premiere * self.seconde)

    def en_serie(self, *, aidant: bool) -> float:
        """`L1 + L2 +- 2M` selon que les enroulements s'aident ou s'opposent."""
        signe = 1 if aidant else -1
        return self.premiere + self.seconde + signe * 2 * self.mutuelle

    def est_parfait(self) -> bool:
        """Vrai si tout le flux de l'un traverse l'autre (`k = 1`)."""
        return math.isclose(self.coefficient(), 1.0)
