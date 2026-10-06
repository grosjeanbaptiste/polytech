"""Exercice cote 2018, question 5 : le dephasage de deux courants couples.

Les sens sont ceux de la figure 5.1 : les deux courants entrent par le
haut ; `u1` est flechee vers le haut, `u2` vers le bas ; le point de `L1`
est en haut, celui de `L2` en bas.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

T: Final = TEMPS
I1, I2 = sp.symbols("I_1 I_2")
"""Les substituts complexes des deux courants."""
L12, W = sp.symbols("L_12 omega", positive=True)
THETA = sp.Symbol("theta", real=True)
AMPLITUDES: Final = sp.symbols("A_1 A_2", positive=True)


def tensions_mutuelles() -> tuple[sp.Expr, sp.Expr]:
    """`U12 = -j omega L12 I2` et `U21 = +j omega L12 I1`."""
    return -sp.I * W * L12 * I2, sp.I * W * L12 * I1


def rapport_des_tensions() -> sp.Expr:
    """`U21 / U12`."""
    u12, u21 = tensions_mutuelles()
    return sp.simplify(u21 / u12)


def dephasage() -> sp.Expr:
    """L'avance de `i1` sur `i2`.

    `U21/U12 = e^(j theta)` par la mesure, et `I1/I2` n'en differe que
    par un facteur reel negatif, dont l'argument est `pi`.
    """
    facteur = sp.simplify((I1 / I2) / rapport_des_tensions())
    return THETA + sp.arg(facteur)


def courant(rang: int) -> sp.Expr:
    """Les deux courants dans le temps, `i1` en avance de `dephasage()` sur `i2`."""
    avance = dephasage() if rang == 1 else sp.Integer(0)
    return AMPLITUDES[rang - 1] * sp.cos(W * T + avance)


def tensions_mutuelles_instantanees() -> tuple[sp.Expr, sp.Expr]:
    """`u12` et `u21` dans le temps, par les equations des bobines couplees.

    `i2` entre dans `L2` par sa borne non pointee : son flux s'oppose a
    celui de `i1` dans `L1`, d'ou le signe moins de `u12`. La fleche de
    `u2` est a l'envers du sens recepteur, ce qui retourne `u21`.
    """
    return -L12 * courant(2).diff(T), L12 * courant(1).diff(T)


def u21_est_u12_avancee_de(angle: sp.Expr) -> bool:
    """Vrai si `u21`, a l'amplitude pres, est `u12` avancee de cet angle."""
    u12, u21 = tensions_mutuelles_instantanees()
    premiere, seconde = AMPLITUDES
    avancee = u12.subs(T, T + angle / W) / seconde
    return bool(sp.simplify(u21 / premiere - avancee) == 0)
