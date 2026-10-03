"""Deux lectures d'oscilloscope sur un circuit du premier ordre.

**Le temps de montee.** Une charge `1 - e^{-t/tau}` atteint 10 % de sa
valeur finale en `tau ln(10/9)` et 90 % en `tau ln 10` : l'intervalle
vaut `tau ln 9`, soit environ `2,2 tau`. La decharge, symetrique, met le
meme temps a descendre de 90 % a 10 %. C'est ce que l'oscilloscope
appelle *rise time* et *fall time*.

**Deux points d'une decharge.** `v(t) = V e^{-t/tau}` donne
`v1 / v2 = e^{(t2 - t1)/tau}`, donc `tau = (t2 - t1) / ln(v1 / v2)`. La
formule n'a de sens que si `t2 > t1` et `v1 > v2` : sinon ce n'est pas
une decharge, et la valeur rendue serait negative ou infinie.
"""

from __future__ import annotations

import math

import sympy as sp


def temps_de_montee(constante_de_temps: sp.Expr) -> sp.Expr:
    """`tau ln 9` : de 10 % a 90 % de la valeur finale."""
    return constante_de_temps * sp.log(9)


def constante_par_deux_points(
    premier_instant: float,
    premiere_tension: float,
    second_instant: float,
    seconde_tension: float,
) -> float:
    """`(t2 - t1) / ln(v1 / v2)` ; leve si les points ne decrivent pas une decharge."""
    if second_instant <= premier_instant:
        msg = f"instants dans le mauvais ordre : {premier_instant}, {second_instant}"
        raise ValueError(msg)
    if not premiere_tension > seconde_tension > 0:
        msg = f"pas une decharge : {premiere_tension} puis {seconde_tension}"
        raise ValueError(msg)
    return (second_instant - premier_instant) / math.log(
        premiere_tension / seconde_tension
    )
