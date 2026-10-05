"""Une grandeur sinusoidale, son substitut complexe, son dephasage.

Le syllabus d'electricite generale (16.2 a 16.5).
`x = X racine(2) cos(omega t + theta_x)` (16.3.7) : `X` est la valeur
efficace, `X racine(2)` l'amplitude (16.3.6), `theta_x` la phase initiale,
prise entre `-pi` et `pi`. La periode vaut `2 pi / omega` (16.2.2).

Son substitut complexe est `X exp(j theta_x)` (16.4.3) ; la grandeur en
est la partie reelle, apres multiplication par `racine(2) exp(j omega t)`
(16.4.6). Deriver revient a multiplier par `j omega` (16.4.7), donc a
tourner d'un quart de tour en avant.

Le dephasage de `y` sur `x` est `theta_y - theta_x`, ramene entre `-pi`
et `pi` (16.5.1) : nul, les grandeurs sont en phase ; `pi / 2` en valeur
absolue, en quadrature ; `pi`, en opposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


class RelationDePhase(Enum):
    """Les trois cas que le syllabus nomme, et tous les autres."""

    EN_PHASE = "en phase"
    EN_QUADRATURE = "en quadrature"
    EN_OPPOSITION = "en opposition"
    QUELCONQUE = "quelconque"


@dataclass(frozen=True, slots=True)
class GrandeurSinusoidale:
    """La valeur efficace, la phase initiale et la pulsation."""

    efficace: Grandeur
    phase: Grandeur
    pulsation: Grandeur

    def __post_init__(self) -> None:
        """Refuse une valeur efficace negative, une phase hors de `[-pi ; pi]`."""
        if sp.sympify(self.efficace).is_negative:
            msg = f"valeur efficace negative : {self.efficace}"
            raise ValueError(msg)
        if (sp.Abs(self.phase) > sp.pi) == sp.true:
            msg = f"phase initiale hors de [-pi ; pi] : {self.phase}"
            raise ValueError(msg)
        if sp.sympify(self.pulsation).is_positive is not True:
            msg = f"pulsation nulle ou negative : {self.pulsation}"
            raise ValueError(msg)

    def periode(self) -> sp.Expr:
        """`T = 2 pi / omega` (16.2.2)."""
        return 2 * sp.pi / self.pulsation

    def frequence(self) -> sp.Expr:
        """`f = 1 / T` (16.2.3)."""
        return 1 / self.periode()

    def amplitude(self) -> sp.Expr:
        """`x_m = X racine(2)` (16.3.6)."""
        return self.efficace * sp.sqrt(2)

    def instantanee(self) -> sp.Expr:
        """`x = X racine(2) cos(omega t + theta_x)` (16.3.7)."""
        return self.amplitude() * sp.cos(self.pulsation * TEMPS + self.phase)

    def substitut(self) -> sp.Expr:
        """`X exp(j theta_x)` (16.4.3)."""
        return self.efficace * sp.exp(sp.I * self.phase)


def substitut_de_la_derivee(
    substitut: sp.Expr, pulsation: Grandeur, ordre: int = 1
) -> sp.Expr:
    """`(j omega)^n X` : le substitut de la derivee d'ordre `n` (16.4.7)."""
    return (sp.I * pulsation) ** ordre * substitut


def dephasage(phase: Grandeur, *, sur: Grandeur) -> sp.Expr:
    """`theta_y - theta_x + 2 k pi`, ramene dans `]-pi ; pi]` (16.5.1)."""
    ecart = sp.sympify(phase) - sur
    tours = sp.ceiling((ecart - sp.pi) / (2 * sp.pi))
    return sp.simplify(ecart - 2 * sp.pi * tours)


def avance(ecart: Grandeur, pulsation: Grandeur) -> sp.Expr:
    """Le dephasage en unite de temps : `theta / omega`."""
    return sp.sympify(ecart) / pulsation


def relation_de_phase(ecart: Grandeur) -> RelationDePhase:
    """En phase, en quadrature, en opposition, ou rien de tout cela."""
    nommes = {
        sp.Integer(0): RelationDePhase.EN_PHASE,
        sp.pi / 2: RelationDePhase.EN_QUADRATURE,
        sp.pi: RelationDePhase.EN_OPPOSITION,
    }
    return nommes.get(sp.Abs(sp.sympify(ecart)), RelationDePhase.QUELCONQUE)
