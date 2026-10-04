"""TP 5 d'electronique, exercices 10 a 15 : les montages a un AOP.

Chaque montage est une liste de composants autour d'un AOP a gain fini
(`domain.aop`) ; le gain s'obtient a la limite du gain infini. Les
resistances sont en ohms, exactes. Le noeud 1 porte toujours la source.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeCourant,
    SourceDeTension,
    resoudre,
)

VIN: Final = sp.Symbol("V_in")
_I: Final = sp.Symbol("I_S")


def _r(ohms: float) -> sp.Expr:
    """La conductance d'une resistance, exacte."""
    return 1 / sp.nsimplify(ohms)


def _gain(circuit: list[Element], noeuds: int, sortie: int) -> float:
    return float(sortie_ideale(circuit, noeuds, sortie) / VIN)


def _inverseur(entree: float, contre_reaction: float) -> list[Element]:
    """Noeud 2 : l'entree inverseuse ; noeud 3 : la sortie."""
    return [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, _r(entree)),
        Admittance(2, 3, _r(contre_reaction)),
        *aop(0, 2, 3),
    ]


def _non_inverseur(vers_la_masse: float, contre_reaction: float) -> list[Element]:
    return [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(2, 0, _r(vers_la_masse)),
        Admittance(2, 3, _r(contre_reaction)),
        *aop(1, 2, 3),
    ]


def gains_de_l_exercice_10() -> tuple[float, float]:
    """L'inverseur, puis le non-inverseur, avec `R1 = 1 k`, `R2 = 10 k`."""
    return _gain(_inverseur(1e3, 10e3), 3, 3), _gain(_non_inverseur(1e3, 10e3), 3, 3)


def gain_de_l_exercice_11() -> float:
    """Le non-inverseur `25 k` sur `5 k`."""
    return _gain(_non_inverseur(5e3, 25e3), 3, 3)


def montage_de_l_exercice_12(r3: float) -> list[Element]:
    """`R3` entre la sortie de l'AOP (noeud 4) et `V_o` (noeud 3)."""
    return [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, _r(1e3)),
        Admittance(2, 3, _r(10e3)),
        Admittance(3, 4, _r(r3)),
        *aop(0, 2, 4),
    ]


def gain_de_l_exercice_12(r3: float) -> float:
    """`V_o / V_in` en regime lineaire : `R3` est dans la boucle."""
    return _gain(montage_de_l_exercice_12(r3), 4, 3)


def exercice_13() -> tuple[float, float]:
    """Le gain, et l'amplitude du courant dans la charge `R3 = 1 k`."""
    circuit = [*_non_inverseur(5e3, 1e3), Admittance(3, 0, _r(1e3))]
    gain = _gain(circuit, 3, 3)
    return gain, 0.5 * gain / 1e3


def exercice_14() -> tuple[float, float]:
    """`V_1 / V_in` : les deux blocs relies directement, puis par un suiveur."""
    direct: list[Element] = [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, _r(1e3)),
        Admittance(2, 0, _r(1e3)),
    ]
    tamponne: list[Element] = [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, _r(1e3)),
        *aop(2, 3, 3),
        Admittance(3, 0, _r(1e3)),
    ]
    return _gain(direct, 2, 2), _gain(tamponne, 3, 3)


def exercice_15() -> tuple[float, float]:
    """Les deux inverseurs precedes d'un reseau resistif."""
    simple: list[Element] = [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, _r(1e3)),
        Admittance(2, 3, _r(1e3)),
        Admittance(3, 4, _r(5e3)),
        *aop(0, 3, 4),
    ]
    thevenin: list[Element] = [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, _r(1e3)),
        Admittance(2, 0, _r(1e3)),
        Admittance(2, 3, _r(500)),
        Admittance(3, 4, _r(1e3)),
        Admittance(4, 5, _r(5e3)),
        *aop(0, 4, 5),
    ]
    return _gain(simple, 4, 4), _gain(thevenin, 5, 5)


def thevenin_de_l_exercice_15() -> tuple[float, float]:
    """Le Thevenin de `{V_in, R3, R4}` : rapport a vide, puis resistance.

    La resistance se mesure source eteinte, en injectant `1 A` au noeud.
    """
    a_vide: list[Element] = [
        SourceDeTension(1, 0, VIN, _I),
        Admittance(1, 2, _r(1e3)),
        Admittance(2, 0, _r(1e3)),
    ]
    eteinte: list[Element] = [
        SourceDeTension(1, 0, sp.Integer(0), _I),
        Admittance(1, 2, _r(1e3)),
        Admittance(2, 0, _r(1e3)),
        SourceDeCourant(0, 2, sp.Integer(1)),
    ]
    return float(resoudre(a_vide, 2)[1] / VIN), float(resoudre(eteinte, 2)[1])
