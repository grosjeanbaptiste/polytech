"""TP 5 d'electronique, exercices 1, 2 et 7 : gains sans AOP ideal.

Le diviseur et l'amplificateur reel sont resolus par l'analyse nodale :
l'amplificateur de l'exercice 2 y est une source `20 V_i` derriere ses
`75 Ohm` de sortie, mise en Norton.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceCommandee,
    SourceDeTension,
    resoudre,
)

_I: Final = sp.Symbol("I_S")


def gain_du_diviseur() -> float:
    """Exercice 1 : `72 kOhm` en haut, `3 kOhm` en bas."""
    vi = sp.Symbol("V_i")
    circuit: list[Element] = [
        SourceDeTension(1, 0, vi, _I),
        Admittance(1, 2, sp.Rational(1, 72_000)),
        Admittance(2, 0, sp.Rational(1, 3_000)),
    ]
    return float(resoudre(circuit, 2)[1] / vi)


def amplificateur_reel() -> tuple[float, float]:
    """Exercice 2 : `V_i` a l'entree de l'amplificateur, et `V_o`."""
    sortie = sp.Rational(1, 75)
    circuit: list[Element] = [
        SourceDeTension(1, 0, sp.Rational(1, 2), _I),
        Admittance(1, 2, sp.Rational(1, 200)),
        Admittance(2, 0, sp.Rational(1, 10_000)),
        SourceCommandee(0, 3, 20 * sortie, 2, 0),
        Admittance(3, 0, sortie),
        Admittance(3, 0, sp.Rational(1, 1_000)),
    ]
    tensions = resoudre(circuit, 3)
    return float(tensions[1]), float(tensions[2])


def sortie_differentielle() -> float:
    """Exercice 7 : un gain de 100 sur la difference des deux entrees."""
    return float(100 * (sp.Rational(183, 10) - sp.Rational(182, 10)))
