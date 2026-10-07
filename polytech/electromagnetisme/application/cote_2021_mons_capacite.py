"""Exercice cote 2021 (Mons), question 3 : charger seize microsecondes, puis decharger.

En position 1 le condensateur se charge a travers `R0` ; en position 2 il
se decharge dans `R`. Les deux constantes de temps sont egales.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.enclenchement import enclenchement
from polytech.electromagnetisme.domain.symboles import TEMPS

T: Final = TEMPS
E0: Final = sp.Integer(10)
R0: Final = sp.Integer(10)
R: Final = sp.Integer(10)
C: Final = sp.Rational(1, 10**6)
BASCULEMENT: Final = sp.Rational(16, 10**6)
"""L'instant ou l'on passe en position 2, en secondes."""


def constante_de_temps() -> sp.Expr:
    """`R0 C`, egale a `R C` : dix microsecondes."""
    return R0 * C


def charge() -> sp.Expr:
    """`u_c(t)` en position 1, le condensateur partant decharge."""
    return enclenchement(E0, sp.Integer(0), constante_de_temps())


def tension_au_basculement() -> sp.Expr:
    """`u_c` a seize microsecondes : la condition initiale de la decharge."""
    return charge().subs(T, BASCULEMENT)


def decharge() -> sp.Expr:
    """`u_c(t)` en position 2 : la tension ne saute pas, puis s'eteint."""
    return tension_au_basculement() * sp.exp(-(T - BASCULEMENT) / (R * C))


def tension(instant: float) -> float:
    """`u_c` a cet instant, en volts, sur l'un ou l'autre morceau."""
    morceau = charge() if instant <= BASCULEMENT else decharge()
    return float(morceau.subs(T, instant))


def energie_dissipee_a_la_decharge() -> sp.Expr:
    """`integrale de u^2/R` du basculement a l'infini."""
    return sp.integrate(decharge() ** 2 / R, (T, BASCULEMENT, sp.oo))
