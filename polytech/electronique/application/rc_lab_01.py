"""Laboratoire 1 d'electronique, section 2.3 : le circuit RC passe-haut.

Le circuit est un diviseur d'impedances : la sortie est prise sur la
resistance, la capacite est dans le bras. Les valeurs sont celles de la
manipulation -- 10 kohm, 10 nF -- et le generateur du Picoscope presente
600 ohm de resistance de sortie.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.domain.abcd import gain_en_tension, serie

RESISTANCE: Final = 10e3
CAPACITE: Final = 10e-9
GENERATEUR: Final = 600.0
"""La resistance de sortie du generateur, lue dans la fiche du Picoscope."""


def passe_haut(resistance: sp.Expr, capacite: sp.Expr, p: sp.Expr) -> sp.Expr:
    """`Z_R / (Z_R + Z_C)` : la capacite en serie, la sortie sur la resistance."""
    return gain_en_tension(serie(1 / (p * capacite)), resistance)


def pulsation_de_coupure() -> float:
    """`w_C = 1 / (R C)` : la ou `|Z_R| = |Z_C|`."""
    return 1 / (RESISTANCE * CAPACITE)


def frequence_de_coupure() -> float:
    """La meme en hertz : c'est ce que l'analyseur de reponse affiche."""
    return pulsation_de_coupure() / (2 * math.pi)


def reponse_a_la_coupure() -> tuple[float, float]:
    """Le module et la phase, en degres, de `H(j w_C)`."""
    p = sp.Symbol("p")
    transfert = passe_haut(sp.nsimplify(RESISTANCE), sp.nsimplify(CAPACITE), p)
    valeur = complex(transfert.subs(p, sp.I * sp.nsimplify(pulsation_de_coupure())))
    return abs(valeur), math.degrees(math.atan2(valeur.imag, valeur.real))


def asymptotes(p: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Le transfert quand `Z_C` domine (`w -> 0`), puis quand `Z_R` domine.

    Aux basses frequences on neglige `R` devant `1 / (p C)` dans la
    somme : `H ~ R / Z_C = p R C`. Aux hautes, l'inverse : `H ~ R / R`.
    """
    produit = sp.nsimplify(RESISTANCE * CAPACITE)
    return p * produit, sp.Integer(1)


def poids_du_generateur() -> float:
    """Question 2 : `R_G / R`, ce que la resistance du generateur pese."""
    return GENERATEUR / RESISTANCE


def effet_du_generateur() -> tuple[float, float]:
    """Le plateau `R / (R + R_G)` et la frequence de coupure avec `R_G`.

    Rapporte a la force electromotrice du generateur, le circuit est
    encore un passe-haut, de resistance `R + R_G` dans sa constante de
    temps et de plateau abaisse par le pont resistif.
    """
    totale = RESISTANCE + GENERATEUR
    return RESISTANCE / totale, 1 / (totale * CAPACITE) / (2 * math.pi)
