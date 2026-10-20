"""Devoir cote de fluides : l'etude parametrique, point par point.

Un point est un fluide, une vitesse d'approche `U` et un diametre `D`. Il
rend ce que le devoir demande de commenter : le Reynolds et son regime,
la vitesse maximale (au sommet du cylindre), l'ecart de pression entre le
point d'arret et le sommet, le Mach dans l'air -- et le coefficient de
pression, calcule en pascals par Bernoulli puis adimensionne, pour montrer
qu'il ne depend d'aucun des quatre parametres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.fluides.application import devoir_cylindre_fluides as dcf
from polytech.fluides.domain import cylindre_potentiel as cyl

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class Point:
    """Un cas de l'etude : un fluide, une vitesse d'approche, un diametre."""

    fluide: dcf.Fluide
    vitesse: float
    diametre: float

    def __post_init__(self) -> None:
        """Refuse une vitesse ou un diametre nuls ou negatifs."""
        if self.vitesse <= 0 or self.diametre <= 0:
            msg = f"vitesse ou diametre non strictement positifs : {self}"
            raise ValueError(msg)

    @property
    def reynolds(self) -> float:
        """`Re = U D / nu`."""
        return dcf.reynolds(self.fluide, self.vitesse, self.diametre)

    @property
    def regime(self) -> dcf.Regime:
        """La zone de la courbe `C_D(Re)` du cours."""
        return dcf.regime(self.reynolds)

    def _vitesse_parietale(self, angle: float) -> float:
        """`-2 U sin(theta)` : sur la paroi, `r = R`, le rayon s'elimine."""
        return float(cyl.vitesse_tangentielle(1, angle, self.vitesse, 1))

    @property
    def vitesse_maximale(self) -> float:
        """`|v_theta|` au sommet, `theta = pi / 2`."""
        return abs(self._vitesse_parietale(math.pi / 2))

    def pression_relative(self, angle: float) -> float:
        """`p - p_inf = rho (U^2 - v^2) / 2` sur la paroi, en pascals."""
        v = self._vitesse_parietale(angle)
        return self.fluide.masse_volumique * (self.vitesse**2 - v**2) / 2

    @property
    def ecart_de_pression_maximal(self) -> float:
        """Du point d'arret au sommet, en pascals."""
        return self.pression_relative(0.0) - self.pression_relative(math.pi / 2)

    def coefficient_de_pression(self, angle: float) -> float:
        """`C_p = (p - p_inf) / (rho U^2 / 2)`, par la pression en pascals."""
        dynamique = self.fluide.masse_volumique * self.vitesse**2 / 2
        return self.pression_relative(angle) / dynamique

    @property
    def mach(self) -> float | None:
        """`U / c` ; sans objet pour l'eau, traitee incompressible."""
        return dcf.mach(self.fluide, self.vitesse)


def balayage_des_vitesses(
    fluide: dcf.Fluide, diametre: float, vitesses: Iterable[float]
) -> list[Point]:
    """`U` varie, `D` et le fluide fixes."""
    return [Point(fluide, u, diametre) for u in vitesses]


def balayage_des_diametres(
    fluide: dcf.Fluide, vitesse: float, diametres: Iterable[float]
) -> list[Point]:
    """`D` varie, `U` et le fluide fixes."""
    return [Point(fluide, vitesse, d) for d in diametres]


def comparaison_des_fluides(vitesse: float, diametre: float) -> tuple[Point, Point]:
    """L'air puis l'eau : `rho` et `mu` changent ensemble."""
    return Point(dcf.AIR, vitesse, diametre), Point(dcf.EAU, vitesse, diametre)


def angles_de_pression_nulle() -> tuple[float, ...]:
    """Les angles ou `C_p = 0` sur `[0, pi]` : `sin^2 = 1/4`."""
    theta = cyl.ANGLE
    racines = sp.solveset(
        cyl.coefficient_de_pression_parietal(theta), theta, sp.Interval(0, sp.pi)
    )
    return tuple(sorted(float(r) for r in racines))
