"""La methode des zones : l'echange total dans une enceinte de surfaces grises.

Syllabus de thermique, tome 1, deuxieme partie, sections 3.1 a 3.6. Une
surface grise renvoie une part de ce qu'elle recoit. Vue de l'exterieur,
elle rayonne son **exitance apparente**, `J = eps sigma T^4 + rho I`, ou
`I` est le flux incident : `S_i I_i = somme_j ij J_j`.

Le flux net d'une zone s'ecrit de deux facons, et les egaler donne un
systeme lineaire en `J` :

    Phi_i = (S_i eps_i / rho_i)(sigma T_i^4 - J_i) = somme_j ij (J_i - J_j).

Une zone noire a `J = sigma T^4`. Une zone **adiabatique** a un flux net
nul : elle renvoie tout, comme un reflecteur parfait, et ni son
emissivite ni sa temperature n'entrent dans le bilan des autres.

L'aire d'echange total `S_iS_j` se lit sur le meme systeme, en portant
la zone `i` seule a un potentiel unite : c'est ce que la zone `j`
absorbe alors, directement ou apres reflexions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from polytech.thermique.domain.rayonnement.corps_noir import emittance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

TOLERANCE = 1e-9


def _exiger_une_emissivite(emissivite: float) -> None:
    if not 0 < emissivite <= 1:
        msg = f"emissivite hors de ]0, 1] : {emissivite}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Zone:
    """Une zone : son aire et son emissivite, ou son caractere adiabatique."""

    aire: float
    emissivite: float
    est_adiabatique: bool = False

    def __post_init__(self) -> None:
        """Refuse une emissivite hors de ]0, 1]."""
        _exiger_une_emissivite(self.emissivite)

    @classmethod
    def adiabatique(cls, aire: float) -> Zone:
        """Une paroi isolee : son emissivite ne joue pas."""
        return cls(aire, 1.0, est_adiabatique=True)


@dataclass(frozen=True, slots=True)
class EnceinteGrise:
    """Des zones, et les aires d'echange direct `ij` entre elles."""

    zones: Sequence[Zone]
    directes: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        """Refuse des aires non symetriques, ou une enceinte non fermee."""
        matrice = np.array(self.directes, dtype=float)
        if not np.allclose(matrice, matrice.T, atol=TOLERANCE):
            msg = f"aires d'echange non symetriques : {self.directes}"
            raise ValueError(msg)
        aires = [zone.aire for zone in self.zones]
        if not np.allclose(matrice.sum(axis=1), aires, atol=TOLERANCE):
            msg = f"enceinte non fermee : {matrice.sum(axis=1)} pour {aires}"
            raise ValueError(msg)

    def _echanges(self) -> NDArray[np.float64]:
        """La matrice qui, appliquee a `J`, rend `somme_j ij (J_i - J_j)`."""
        matrice = np.array(self.directes, dtype=float)
        echanges = np.diag(matrice.sum(axis=1)) - matrice
        return np.asarray(echanges, dtype=float)

    def exitances_apparentes(self, noirs: Sequence[float]) -> list[float]:
        """Les `J_i`, pour les potentiels `sigma T_i^4` donnes."""
        systeme = self._echanges()
        second = np.zeros(len(self.zones))
        for i, zone in enumerate(self.zones):
            if zone.est_adiabatique:
                continue
            systeme[i] *= 1 - zone.emissivite
            surface = zone.aire * zone.emissivite
            systeme[i, i] += surface
            second[i] = surface * noirs[i]
        return [float(j) for j in np.linalg.solve(systeme, second)]

    def _flux(self, noirs: Sequence[float]) -> list[float]:
        apparentes = np.array(self.exitances_apparentes(noirs))
        return [float(f) for f in self._echanges() @ apparentes]

    def flux_nets(self, temperatures: Sequence[float]) -> list[float]:
        """Le flux net que chaque zone fournit a l'enceinte, en watts."""
        return self._flux([emittance(t) for t in temperatures])

    def temperature_d_equilibre(
        self, indice: int, temperatures: Sequence[float]
    ) -> float:
        """La temperature que prend une zone adiabatique : `sigma T^4 = J`."""
        if not self.zones[indice].est_adiabatique:
            msg = f"zone non adiabatique : {indice}"
            raise ValueError(msg)
        noirs = [emittance(t) for t in temperatures]
        apparente = self.exitances_apparentes(noirs)[indice]
        return float((apparente / emittance(1.0)) ** 0.25)

    def aires_d_echange_total(self) -> list[list[float]]:
        """Les `S_iS_j` ; une zone adiabatique n'en porte aucune."""
        nombre = len(self.zones)
        totales = [[0.0] * nombre for _ in range(nombre)]
        actives = [i for i, z in enumerate(self.zones) if not z.est_adiabatique]
        for i in actives:
            seule = [1.0 if k == i else 0.0 for k in range(nombre)]
            flux = self._flux(seule)
            for j in actives:
                totales[i][j] = -flux[j]
            zone = self.zones[i]
            totales[i][i] = zone.aire * zone.emissivite - flux[i]
        return totales
