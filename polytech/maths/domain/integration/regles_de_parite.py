"""Quel changement de variable pour une fonction rationnelle en `sin x` et `cos x`.

La parite de `R(sin x, cos x)` le dit :

* impaire en `sin x` -- `R(-sin, cos) = -R` : poser `u = cos x` ;
* impaire en `cos x` -- `R(sin, -cos) = -R` : poser `u = sin x` ;
* paire en `sin x` et en `cos x` a la fois -- `R(-sin, -cos) = R` : poser
  `t = tan x` ;
* « quand rien d'autre ne convient », `t = tan(x/2)`, qui marche toujours
  mais coute plus cher.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE


class Changement(Enum):
    """Les quatre changements de variable, du plus simple au plus general."""

    COSINUS = "u = cos x"
    SINUS = "u = sin x"
    TANGENTE = "t = tan x"
    ARC_MOITIE = "t = tan(x/2)"


def changement_conseille(integrande: sp.Expr) -> Changement:
    """Le changement de variable que la parite de l'integrande designe."""
    s, c = sp.symbols("s c")
    en_sinus_et_cosinus = sp.sympify(integrande).replace(
        sp.tan, lambda angle: sp.sin(angle) / sp.cos(angle)
    )
    rationnelle = en_sinus_et_cosinus.subs({sp.sin(VARIABLE): s, sp.cos(VARIABLE): c})

    def change(signe_du_sinus: int, signe_du_cosinus: int, signe: int) -> bool:
        image = rationnelle.subs(
            {s: signe_du_sinus * s, c: signe_du_cosinus * c}, simultaneous=True
        )
        return bool(sp.simplify(image - signe * rationnelle) == 0)

    if change(-1, 1, -1):
        return Changement.COSINUS
    if change(1, -1, -1):
        return Changement.SINUS
    return Changement.TANGENTE if change(-1, -1, 1) else Changement.ARC_MOITIE
