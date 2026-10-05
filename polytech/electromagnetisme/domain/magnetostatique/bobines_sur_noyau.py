"""Les inductances propres et mutuelle de deux bobines sur un noyau.

Le syllabus d'electricite generale (10.15). Deux bobines de `N_1` et
`N_2` spires partagent un tube de flux de reluctance `R_12` ; le flux de
chacune emprunte aussi un tube de dispersion, de reluctance `R'`, qui
evite l'autre. La reluctance vue d'une bobine est leur association en
parallele, donc au plus `R_12`.

`L_1 = N_1^2 / R_1` (10.15.2), `L_12 = N_1 N_2 / R_12` (10.15.3) : le
facteur de couplage `k = racine(L_12^2 / (L_1 L_2))` (10.15.5) vaut donc
`racine(R_1 R_2) / R_12` (10.15.6), entre zero et un. Avec dispersion,
`L_1 = (N_1 / N_2) L_12 + l_1` (10.15.11), `l_1` etant l'inductance propre
de dispersion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.electromagnetisme.domain.magnetostatique.reluctances import (
    reluctance_en_parallele,
)


def facteur_de_dispersion(couplage: float) -> float:
    """`sigma = 1 - k^2` (10.15.8)."""
    if not 0 <= couplage <= 1:
        msg = f"facteur de couplage hors de [0 ; 1] : {couplage}"
        raise ValueError(msg)
    return 1 - couplage**2


@dataclass(frozen=True, slots=True)
class BobinesSurNoyau:
    """Deux bobines, leur tube commun et leurs deux tubes de dispersion."""

    spires: tuple[int, int]
    commune: float
    dispersions: tuple[float, float]

    def __post_init__(self) -> None:
        """Refuse une bobine sans spire et une reluctance non positive."""
        for nombre in self.spires:
            if nombre <= 0:
                msg = f"nombre de spires nul ou negatif : {nombre}"
                raise ValueError(msg)
        for valeur in (self.commune, *self.dispersions):
            if valeur <= 0:
                msg = f"reluctance nulle ou negative : {valeur} A/Wb"
                raise ValueError(msg)

    def reluctance_vue(self, rang: int) -> float:
        """`R_1` ou `R_2` : le tube commun et la dispersion en parallele."""
        return reluctance_en_parallele(self.commune, self.dispersions[rang])

    def propre(self, rang: int) -> float:
        """`L = N^2 / R` (10.15.2) et (10.15.4), en henrys."""
        return self.spires[rang] ** 2 / self.reluctance_vue(rang)

    def mutuelle(self) -> float:
        """`L_12 = N_1 N_2 / R_12` (10.15.3), en henrys."""
        return self.spires[0] * self.spires[1] / self.commune

    def facteur_de_couplage(self) -> float:
        """`k = racine(R_1 R_2) / R_12` (10.15.6)."""
        vues = self.reluctance_vue(0) * self.reluctance_vue(1)
        return math.sqrt(vues) / self.commune

    def de_dispersion(self, rang: int) -> float:
        """`l = L - (N / N') L_12` (10.15.11), en henrys."""
        rapport = self.spires[rang] / self.spires[1 - rang]
        return self.propre(rang) - rapport * self.mutuelle()
