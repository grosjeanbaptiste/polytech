"""Les proprietes de l'air sec a basse pression.

Syllabus de thermique, tome 2, table 3.2 (« Properties of Dry Air at Low
Pressure »), lignes de 250 a 500 K. Les unites de la table --
kJ/(kg K), 1e-5 N s/m^2, 1e-5 kW/(m K) -- sont ramenees au systeme
international. La masse volumique est celle a une atmosphere.

Les colonnes `gamma` et `nu` de la table ne sont pas recopiees : elles
se deduisent des autres, et un test les confronte aux valeurs imprimees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polytech.fluides.domain.proprietes import viscosite_cinematique
from polytech.thermique.domain.tables.interpolation import interpoler


@dataclass(frozen=True, slots=True)
class AirSec:
    """Une ligne : `T` en K, `c_p`, `c_v`, `mu`, `lambda`, `Pr`, `rho`."""

    temperature: float
    chaleur_massique: float
    chaleur_a_volume_constant: float
    viscosite: float
    conductivite: float
    prandtl: float
    masse_volumique: float

    def gamma(self) -> float:
        """`c_p / c_v`."""
        return self.chaleur_massique / self.chaleur_a_volume_constant

    def viscosite_cinematique(self) -> float:
        """`nu = mu / rho`, a une atmosphere."""
        return viscosite_cinematique(self.viscosite, self.masse_volumique)


TABLE_DE_L_AIR_SEC: Final = (
    AirSec(250, 1003.1, 716.0, 1.599e-5, 2.227e-2, 0.720, 1.412),
    AirSec(275, 1003.8, 716.7, 1.725e-5, 2.428e-2, 0.713, 1.284),
    AirSec(300, 1004.9, 717.8, 1.846e-5, 2.624e-2, 0.707, 1.177),
    AirSec(325, 1006.3, 719.2, 1.962e-5, 2.816e-2, 0.701, 1.086),
    AirSec(350, 1008.2, 721.1, 2.075e-5, 3.003e-2, 0.697, 1.009),
    AirSec(375, 1010.6, 723.5, 2.181e-5, 3.186e-2, 0.692, 0.9413),
    AirSec(400, 1013.5, 726.4, 2.286e-5, 3.365e-2, 0.688, 0.8824),
    AirSec(450, 1020.6, 733.5, 2.485e-5, 3.710e-2, 0.684, 0.7844),
    AirSec(500, 1029.5, 742.4, 2.670e-5, 4.041e-2, 0.680, 0.7060),
)


def air_sec(temperature: float) -> AirSec:
    """L'air sec a cette temperature, en kelvins."""
    return interpoler(TABLE_DE_L_AIR_SEC, temperature)
