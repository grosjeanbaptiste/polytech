"""Le reservoir qui se vide par gravite : un modele non lineaire, linearise.

Planches « Systemes » du cours de signaux et systemes, exemple 3. Le
bilan de volume d'un reservoir de section `S`, alimente par `q_e` et qui
se vide par `q_s = alpha racine(h)`, donne le modele global

`S dh/dt = q_e - alpha racine(h)`,

non lineaire, valable partout. A debit d'entree constant, le niveau se
fixe en `h^0 = (q_e^0 / alpha)^2`. En variations autour de ce point, et
en ne gardant que le premier terme du developpement de la racine, il
reste le modele local

`S d(Delta h)/dt = Delta q_e - alpha Delta h / (2 racine(h^0))`,

lineaire, de transmittance `1 / (p S + alpha / (2 racine(h^0)))` -- mais
qui ne vaut qu'autour du point choisi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.transmittance import Transmittance

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Reservoir:
    """La section `S` et le coefficient `alpha` de l'orifice."""

    section: float
    coefficient: float

    def derivee(self, hauteur: float, debit: float) -> float:
        """`dh/dt = (q_e - alpha racine(h)) / S` : le modele global."""
        sortie = self.coefficient * math.sqrt(max(hauteur, 0.0))
        return (debit - sortie) / self.section

    def hauteur_de_repos(self, debit: float) -> float:
        """`h^0 = (q_e^0 / alpha)^2`."""
        if debit <= 0:
            msg = f"debit nul ou negatif : {debit} -- le reservoir se vide"
            raise ValueError(msg)
        return (debit / self.coefficient) ** 2

    def pente_de_sortie(self, debit: float) -> float:
        """`alpha / (2 racine(h^0))`, soit `alpha^2 / (2 q_e^0)`."""
        return self.coefficient / (2 * math.sqrt(self.hauteur_de_repos(debit)))

    def modele_local(self, debit: float) -> Transmittance:
        """`Delta H / Delta Q_e = 1 / (p S + alpha / (2 racine(h^0)))`."""
        return Transmittance(
            expression=1 / (P * self.section + self.pente_de_sortie(debit))
        )

    def premier_ordre_local(self, debit: float) -> tuple[float, float]:
        """Le gain statique et la constante de temps du modele local."""
        pente = self.pente_de_sortie(debit)
        return (1 / pente, self.section / pente)

    def simuler(
        self,
        debit: Callable[[float], float],
        *,
        depart: float,
        instants: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """`h(t)` par le modele global, depuis le repos du debit `depart`."""
        initiale = [self.hauteur_de_repos(depart)]
        solution = solve_ivp(
            lambda t, h: [self.derivee(h[0], debit(t))],
            (float(instants[0]), float(instants[-1])),
            initiale,
            t_eval=instants,
            rtol=1e-9,
            atol=1e-12,
        )
        return np.asarray(solution.y[0], dtype=float)
