"""La resistance et la conductance d'un troncon de tube de courant.

Le syllabus d'electricite generale (13.8 et 13.9). La loi d'Ohm integree
le long d'un tube donne `R_12 = integrale de rho dl / S` et son inverse
`G_12` (13.8.9). Pour un conducteur homogene, `R = rho l / S` (13.9.5) --
l'expression de la reluctance (10.11.12), ou la conductivite prend la
place de la permeabilite.

Deux decoupages servent aux formes compliquees : des tranches en serie,
`dR = rho dl / S` (13.9.6), ou des tubes en parallele,
`dG = dS / (rho l)` (13.9.7).

Les conductances s'associent a l'inverse des resistances : leurs inverses
s'ajoutent en serie (13.9.3), elles s'ajoutent en parallele (13.9.4).
"""

from __future__ import annotations

import math

import sympy as sp


def _exiger_la_resistivite(resistivite: float) -> None:
    if resistivite <= 0:
        msg = f"resistivite nulle ou negative : {resistivite} ohm.m"
        raise ValueError(msg)


def resistance_uniforme(resistivite: float, longueur: float, section: float) -> float:
    """`R = rho l / S` (13.9.5), en ohms."""
    _exiger_la_resistivite(resistivite)
    if longueur <= 0 or section <= 0:
        msg = f"geometrie invalide : longueur {longueur} m, section {section} m^2"
        raise ValueError(msg)
    return resistivite * longueur / section


def resistance_d_un_tube(
    section: sp.Expr, abscisse: sp.Symbol, longueur: object, resistivite: object
) -> sp.Expr:
    """`integrale de 0 a l de rho ds / S(s)` : (13.9.6) en serie, en ohms."""
    integrande = sp.sympify(resistivite) / section
    return sp.simplify(sp.integrate(integrande, (abscisse, 0, longueur)))


def conductance_d_un_arc(
    resistivite: float,
    *,
    hauteur: float,
    interieur: float,
    exterieur: float,
    angle: float,
) -> float:
    """`G = h ln(r_2 / r_1) / (rho theta)` : (13.9.7) en parallele, en siemens."""
    _exiger_la_resistivite(resistivite)
    if not 0 < interieur < exterieur:
        msg = f"rayons invalides : interieur {interieur} m, exterieur {exterieur} m"
        raise ValueError(msg)
    if angle <= 0:
        msg = f"angle nul ou negatif : {angle} rad"
        raise ValueError(msg)
    return hauteur * math.log(exterieur / interieur) / (resistivite * angle)


def _positives(conductances: tuple[float, ...]) -> tuple[float, ...]:
    for valeur in conductances:
        if valeur <= 0:
            msg = f"conductance nulle ou negative : {valeur} S"
            raise ValueError(msg)
    return conductances


def conductance_en_serie(*conductances: float) -> float:
    """`1 / G = somme des 1 / G_k` (13.9.3)."""
    return 1 / sum(1 / valeur for valeur in _positives(conductances))


def conductance_en_parallele(*conductances: float) -> float:
    """`G = somme des G_k` (13.9.4)."""
    return sum(_positives(conductances))
