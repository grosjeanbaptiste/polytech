"""Laboratoire 5 d'electronique, section 2.2.6 : le controle basses et aigus.

Deux voies en parallele derriere l'etage d'entree : un passe-bas RC suivi
d'un inverseur a gain reglable (les basses), un passe-haut RC suivi du
meme inverseur (les aigus). Chaque filtre est charge par les 100 kohm
d'entree de son inverseur, ce qui deplace sa coupure et abaisse le
plateau des basses.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import sympy as sp

from polytech.electronique.domain.aop_ideal import tensions_a_aop_ideaux
from polytech.electronique.domain.netliste import Admittance

if TYPE_CHECKING:
    from collections.abc import Callable

R3: Final = 2.7e3
R4: Final = 100e3
R5: Final = 100e3
C1: Final = 100e-9
"""La voie des basses : `R_3` et `C_1` filtrent, `R_4` et `R_5` amplifient."""
R6: Final = 2.7e3
R7: Final = 100e3
R8: Final = 100e3
C2: Final = 100e-9
"""La voie des aigus : `C_2` et `R_6` filtrent, `R_7` et `R_8` amplifient."""
_P: Final = sp.Symbol("p")


def frequence_de_separation() -> float:
    """La moyenne geometrique du Do0 et du Do10 : le Do5."""
    return math.sqrt(16.35 * 16742)


def _g(resistance: float) -> sp.Expr:
    return 1 / sp.nsimplify(resistance, rational=True)


def basses(p: sp.Expr, *, potentiometre: float = R5) -> sp.Expr:
    """`R_3` en serie, `C_1` a la masse, puis l'inverseur `R_4`, `R_5`.

    Noeuds : 1 l'entree, 2 entre `R_3`, `C_1` et `R_4`, 3 l'entree `-`,
    4 la sortie.
    """
    branches = [Admittance(1, 2, _g(R3)), Admittance(2, 0, p * sp.nsimplify(C1))]
    branches += [Admittance(2, 3, _g(R4)), Admittance(3, 4, _g(potentiometre))]
    return tensions_a_aop_ideaux(branches, [(0, 3, 4)], {1: 1}, 4)[3]


def aigus(p: sp.Expr, *, potentiometre: float = R8) -> sp.Expr:
    """`C_2` en serie, `R_6` a la masse, puis l'inverseur `R_7`, `R_8`."""
    branches = [Admittance(1, 2, p * sp.nsimplify(C2)), Admittance(2, 0, _g(R6))]
    branches += [Admittance(2, 3, _g(R7)), Admittance(3, 4, _g(potentiometre))]
    return tensions_a_aop_ideaux(branches, [(0, 3, 4)], {1: 1}, 4)[3]


def plateau_des_basses() -> float:
    """Le gain en continu, potentiometre au maximum : `R_5 / (R_3 + R_4)`."""
    return R5 / (R3 + R4)


def coupures_des_basses() -> tuple[float, float]:
    """La frequence ou le module vaut `1/racine(2)`, puis celle a -3 dB du plateau.

    La premiere est celle du corrige officiel ; la seconde, `1 / (2 pi
    C_1 (R_3 // R_4))`, est la definition usuelle, et celle de l'enonce.
    """
    produit = C1 * R3 * R4
    absolue = math.sqrt(2 * R4**2 - (R3 + R4) ** 2) / produit
    return absolue / (2 * math.pi), (R3 + R4) / produit / (2 * math.pi)


def coupure_des_aigus() -> float:
    """`1 / (2 pi C_2 (R_6 // R_7))`."""
    return (R6 + R7) / (C2 * R6 * R7) / (2 * math.pi)


@functools.cache
def _numerique(voie: str, potentiometre: float) -> Callable[[Any], Any]:
    """Une voie, ou la somme des deux, comme fonction numerique de `p`."""
    if voie == "basses":
        transfert = basses(_P, potentiometre=potentiometre)
    elif voie == "aigus":
        transfert = aigus(_P, potentiometre=potentiometre)
    else:
        transfert = basses(_P) + aigus(_P)
    fonction: Callable[[Any], Any] = sp.lambdify(_P, transfert, "numpy")
    return fonction


def _module(voie: str, frequence: float, potentiometre: float = R5) -> float:
    return float(abs(_numerique(voie, potentiometre)(2j * math.pi * frequence)))


def gain_des_basses(frequence: float, *, potentiometre: float = R5) -> float:
    """Le module du gain de la voie des basses a cette frequence."""
    if potentiometre == 0:
        return 0.0
    return _module("basses", frequence, potentiometre)


def gain_des_aigus(frequence: float, *, potentiometre: float = R8) -> float:
    """Le module du gain de la voie des aigus a cette frequence."""
    if potentiometre == 0:
        return 0.0
    return _module("aigus", frequence, potentiometre)


def somme_au_maximum(frequence: float) -> float:
    """Le module de la somme des deux voies, les deux reglages au maximum."""
    return _module("somme", frequence)


def ondulation_de_la_somme() -> float:
    """L'ecart, en decibels, entre le haut et le bas de cette somme."""
    return -20 * math.log10(plateau_des_basses())


def courbes() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Les frequences, puis le gain en decibels des basses, des aigus, de la somme."""
    frequences = np.logspace(math.log10(20.0), math.log10(20e3), 181)
    points = 2j * math.pi * frequences
    en_db = [
        20 * np.log10(np.abs(_numerique(voie, R5)(points)))
        for voie in ("basses", "aigus", "somme")
    ]
    return frequences, en_db[0], en_db[1], en_db[2]
