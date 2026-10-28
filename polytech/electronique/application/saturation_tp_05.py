"""TP 5 d'electronique : quand la sortie de l'AOP touche ses rails.

Exercice 8 : en boucle ouverte, un gain de `10^5` sature des que
l'entree differentielle depasse `120 uV`. Exercice 12 : la sortie de
l'AOP n'est pas `V_o` ; elle porte en plus la chute dans `R_3`, et c'est
elle qui sature la premiere.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.application.montages_tp_05 import (
    VIN,
    gains_de_l_exercice_10,
    montage_de_l_exercice_12,
)
from polytech.electronique.domain.aop import saturer, sortie_ideale, sortie_saturee

ALIMENTATION: Final = 12.0
_BOUCLE_OUVERTE: Final = 1e5


def _sinus(t: float) -> float:
    return math.sin(2 * math.pi * 1000 * t)


def entree_differentielle_du_cas_c(instant: float) -> float:
    """`V+ - V-` du cas c : `0 - sin(2 pi 1000 t)`."""
    return 0.0 - _sinus(instant)


def sortie_du_cas_c(instant: float) -> float:
    """La sortie du cas c a cet instant : carre entre les deux rails."""
    return saturer(
        _BOUCLE_OUVERTE * entree_differentielle_du_cas_c(instant), ALIMENTATION
    )


def boucle_ouverte() -> dict[str, float]:
    """Les six cas, lus a la crete du sinus (`t = 0.25 ms`).

    Les cas d et e sont lus au creux (`t = 0.75 ms`) : c'est la que
    l'entree differentielle est la plus faible, donc la que la sortie
    aurait une chance de quitter le rail.
    """
    crete, creux = 0.25e-3, 0.75e-3
    entrees = {
        "a": (1e-6, 0.0),
        "b": (0.0, 1.0),
        "c": (0.0, _sinus(crete)),
        "d": (_sinus(creux), -3.0),
        "e": (1e-6 * _sinus(creux), -3.0),
        "f": (1e-6 * _sinus(crete), 0.0),
    }
    return {
        cas: saturer(_BOUCLE_OUVERTE * (plus - moins), ALIMENTATION)
        for cas, (plus, moins) in entrees.items()
    }


def exercice_10b() -> tuple[float, float]:
    """`V_in = -4 V` : l'inverseur vise `+40 V`, le non-inverseur `-44 V`."""
    inverseur, non_inverseur = gains_de_l_exercice_10()
    return saturer(-4 * inverseur, ALIMENTATION), saturer(
        -4 * non_inverseur, ALIMENTATION
    )


def sortie_de_l_aop_12(r3: float) -> float:
    """`V_s / V_in` : la sortie de l'AOP, en amont de `R3`."""
    return float(sortie_ideale(montage_de_l_exercice_12(r3), 4, 4) / VIN)


def seuil_de_l_exercice_12(r3: float) -> float:
    """L'amplitude d'entree a partir de laquelle la sortie de l'AOP sature."""
    return ALIMENTATION / abs(sortie_de_l_aop_12(r3))


def crete_de_l_exercice_12(r3: float) -> float:
    """`V_o` a `V_in = -2 V`, sortie de l'AOP collee a `+12 V`."""
    tensions = sortie_saturee(montage_de_l_exercice_12(r3), 4, 4, sp.Integer(12))
    return float(tensions[2].subs(VIN, -2))
