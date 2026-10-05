"""La charge electrique, et les particules qui la portent.

Le syllabus d'electricite generale (1.2 et tableau 1.3.1) : « la charge
electrique d'un systeme ne peut varier que par multiples entiers d'une
charge elementaire e », `Q = N e` avec `N` entier, positif ou negatif. Elle
est toujours liee a la matiere : l'electron porte `-e`, le proton `+e`, et
le proton est 1 836 fois plus lourd que l'electron.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.constantes import CHARGE_ELEMENTAIRE

TOLERANCE_SUR_LE_MULTIPLE: Final = 1e-9
"""Ecart admis entre `Q / e` et l'entier le plus proche."""


@dataclass(frozen=True, slots=True)
class Particule:
    """Une ligne du tableau 1.3.1 : masse au repos, charge, moment, spin."""

    masse: float
    charges: int
    moment_magnetique: float
    spin: sp.Rational

    @property
    def charge(self) -> float:
        """La charge en coulombs : un nombre entier de charges elementaires."""
        return charge_d_un_systeme(self.charges)


def charge_d_un_systeme(nombre: float) -> float:
    """`Q = N e`, pour un nombre entier `N` de charges elementaires."""
    if nombre != int(nombre):
        msg = f"un nombre entier de charges elementaires est attendu, pas {nombre}"
        raise TypeError(msg)
    return nombre * CHARGE_ELEMENTAIRE


def nombre_de_charges_elementaires(charge: float) -> int:
    """Le nombre entier `N` tel que `Q = N e` ; leve si la charge n'en est pas un."""
    quotient = charge / CHARGE_ELEMENTAIRE
    if abs(quotient - round(quotient)) > TOLERANCE_SUR_LE_MULTIPLE:
        msg = f"la charge {charge} C n'est pas un multiple entier de e"
        raise ValueError(msg)
    return round(quotient)


_DEMI: Final = sp.Rational(1, 2)

PARTICULES: Final = {
    "proton": Particule(1.673e-27, 1, 1.41e-26, _DEMI),
    "electron": Particule(9.109e-31, -1, 9.27e-24, _DEMI),
    "neutron": Particule(1.675e-27, 0, -9.64e-27, _DEMI),
    "photon": Particule(0.0, 0, 0.0, sp.Integer(1)),
}
"""Tableau 1.3.1 : masse (kg), charge (en e), moment magnetique (A.m^2), spin."""
