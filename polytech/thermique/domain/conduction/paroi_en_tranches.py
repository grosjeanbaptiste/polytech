"""La paroi plane decoupee en tranches : le schema en pi.

Syllabus de thermique, tome 1, premiere partie, section 5.2.4. Une paroi
se decoupe en tranches d'epaisseur `b`. Chacune est une resistance
`R = b / lambda` entre deux demi-capacites `C / 2`, avec `C = rho c b` :
un noeud interieur porte `C`, un noeud de surface `C / 2`, et la
jonction de deux materiaux la demi-somme des leurs.

Le bilan de chaque noeud donne une equation differentielle ordinaire ;
aux deux surfaces, l'echange avec le fluide s'y ajoute. Le systeme
s'integre par un schema implicite, stable quel que soit le pas. Le flux
recu a l'intrados, integre sur la duree, est l'energie depensee pour
chauffer la paroi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Tranches:
    """Une couche : epaisseur, conductivite, `rho c`, et son nombre de tranches."""

    epaisseur: float
    conductivite: float
    capacite: float
    nombre: int

    def resistance(self) -> float:
        """`R = b / lambda`, pour une tranche."""
        return self.epaisseur / (self.nombre * self.conductivite)

    def capacite_de_tranche(self) -> float:
        """`C = rho c b`, pour une tranche."""
        return self.capacite * self.epaisseur / self.nombre


@dataclass(frozen=True, slots=True)
class ParoiEnTranches:
    """Des couches, de l'intrados vers l'extrados, entre deux fluides."""

    couches: Sequence[Tranches]
    interieur: float
    exterieur: float

    def _resistances(self) -> NDArray[np.float64]:
        return np.array([c.resistance() for c in self.couches for _ in range(c.nombre)])

    def capacites(self) -> NDArray[np.float64]:
        """La capacite de chaque noeud, une demi-tranche de chaque cote."""
        tranches = [
            c.capacite_de_tranche() for c in self.couches for _ in range(c.nombre)
        ]
        demis = np.array(tranches) / 2
        return np.concatenate(([demis[0]], demis[:-1] + demis[1:], [demis[-1]]))

    def resistance_totale(self) -> float:
        """Les deux echanges et toutes les tranches, en serie."""
        surfaces = 1 / self.interieur + 1 / self.exterieur
        return float(surfaces + self._resistances().sum())

    def derivees(
        self, temperatures: NDArray[np.float64], *, four: float, ambiance: float
    ) -> NDArray[np.float64]:
        """`dt_i/dtau` en chaque noeud."""
        bilans = np.zeros(len(temperatures))
        traversant = (temperatures[:-1] - temperatures[1:]) / self._resistances()
        bilans[:-1] -= traversant
        bilans[1:] += traversant
        bilans[0] += self.interieur * (four - temperatures[0])
        bilans[-1] += self.exterieur * (ambiance - temperatures[-1])
        return np.asarray(bilans / self.capacites(), dtype=float)

    def energie_fournie(
        self, four: Callable[[float], float], *, ambiance: float, duree: float
    ) -> float:
        """`Q_pi`, en J/m^2 : l'integrale du flux recu a l'intrados."""
        noeuds = len(self.capacites())

        def systeme(instant: float, etat: NDArray[np.float64]) -> list[float]:
            pentes = self.derivees(etat[:-1], four=four(instant), ambiance=ambiance)
            recu = self.interieur * (four(instant) - etat[0])
            return [*pentes, recu]

        depart = [*([ambiance] * noeuds), 0.0]
        solution = solve_ivp(
            systeme, (0.0, duree), depart, method="BDF", rtol=1e-8, max_step=60.0
        )
        return float(solution.y[-1, -1])
