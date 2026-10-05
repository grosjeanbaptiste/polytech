"""La branche fondamentale des reseaux a courants quasi-stationnaires.

Le syllabus d'electricite generale (15.3 a 15.6). Un troncon de tube de
courant qui porte une source, une resistance, une bobine et un
condensateur suit `u = e + R i + L di/dt + q / C` (15.5.8), avec
`i = dq/dt` (15.5.9) ; une bobine voisine y ajoute `M di'/dt` (15.5.10).
Les equations de Kirchhoff et les theoremes des reseaux lineaires
subsistent.

La puissance absorbee `u i` se partage (15.6.1) entre la source, l'effet
Joule, et les accroissements des energies `L i^2 / 2` et `C u_C^2 / 2`.
Si la bobine ou le condensateur se deforme, un terme de plus apparait :
la puissance que les forces du champ developpent (15.6.2) a (15.6.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


@dataclass(frozen=True, slots=True)
class BrancheFondamentale:
    """`e`, `R`, `L` et `C` ; sans condensateur, la capacite est infinie."""

    fem: Any
    resistance: Any
    inductance: Any = 0
    capacite: Any = sp.oo

    def __post_init__(self) -> None:
        """Refuse une capacite nulle : la tension n'y serait pas definie."""
        if self.capacite == 0:
            msg = "capacite nulle : q / C n'est pas defini"
            raise ValueError(msg)

    def tension(
        self, charge: sp.Expr, *, mutuelle: Grandeur = 0, voisin: Grandeur = 0
    ) -> sp.Expr:
        """`u = e + R i + L di/dt + M di'/dt + q / C` (15.5.10)."""
        courant = sp.diff(charge, TEMPS)
        induite = self.inductance * sp.diff(courant, TEMPS)
        couplee = mutuelle * sp.diff(sp.sympify(voisin), TEMPS)
        ohmique = self.fem + self.resistance * courant
        return ohmique + induite + couplee + charge / self.capacite

    def puissances(self, charge: sp.Expr) -> tuple[sp.Expr, ...]:
        """Les quatre termes de (15.6.1) : source, Joule, bobine, condensateur."""
        courant = sp.diff(charge, TEMPS)
        magnetique = sp.diff(self.inductance * courant**2 / 2, TEMPS)
        electrique = sp.diff(charge**2 / (2 * self.capacite), TEMPS)
        return (
            self.fem * courant,
            self.resistance * courant**2,
            magnetique,
            electrique,
        )


def scinder_la_puissance(coefficient: sp.Expr, grandeur: sp.Expr) -> tuple[Any, Any]:
    """`(d/dt (X y^2 / 2), (1/2) y^2 dX/dt)` : (15.6.2) pour `L, i` ; (15.6.3)."""
    emmagasinee = sp.diff(coefficient * grandeur**2 / 2, TEMPS)
    return emmagasinee, grandeur**2 * sp.diff(coefficient, TEMPS) / 2


def scinder_la_puissance_mutuelle(
    mutuelle: sp.Expr, premier: sp.Expr, second: sp.Expr
) -> tuple[Any, Any]:
    """`(d/dt (M i i'), i i' dM/dt)` (15.6.4)."""
    emmagasinee = sp.diff(mutuelle * premier * second, TEMPS)
    return emmagasinee, premier * second * sp.diff(mutuelle, TEMPS)
