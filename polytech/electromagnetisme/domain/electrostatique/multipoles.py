"""Le developpement multipolaire du potentiel, vu de loin.

Le syllabus d'electricite generale (2.7) : quand le point `P` est tres
eloigne de toutes les charges, `V = V_0 + V_1 + V_2 + ...` (2.7.4), avec

    V_0 = (1 / 4 pi eps_0) Q / r                          (unipolaire),
    V_1 = (1 / 4 pi eps_0) p . u_r / r^2                  (dipolaire),
    V_2 = (1 / 4 pi eps_0 r^3) somme q_i [3/2 (r_i . u_r)^2 - r_i^2 / 2]
                                                          (quadrupolaire).

Un dipole seul place a l'origine a pour potentiel exact `V_1` (2.7.8), et
pour champ `E = [3 (p . r) r / r^5 - p / r^3] / (4 pi eps_0)` (2.7.9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import (
        Point,
        SystemeDeCharges,
    )
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur

DISTANCE: Final = sp.sqrt(sum(axe**2 for axe in CARTESIENNES))
"""`r`, la distance du point courant a l'origine."""


def _scalaire(vecteur: Point) -> sp.Expr:
    return sum(
        (c * axe for c, axe in zip(vecteur, CARTESIENNES, strict=True)), sp.Integer(0)
    )


def contribution_unipolaire(systeme: SystemeDeCharges) -> sp.Expr:
    """`V_0 = Q / (4 pi eps_0 r)` : la charge totale, comme ramenee a l'origine."""
    return COULOMB * systeme.charge_totale() / DISTANCE


def potentiel_du_dipole(moment: Point) -> sp.Expr:
    """`V = p . r / (4 pi eps_0 r^3)`, pour un dipole place a l'origine."""
    return COULOMB * _scalaire(moment) / DISTANCE**3


def contribution_dipolaire(systeme: SystemeDeCharges) -> sp.Expr:
    """`V_1`, le potentiel du moment electrique pris par rapport a l'origine."""
    return potentiel_du_dipole(systeme.moment_electrique())


def contribution_quadrupolaire(systeme: SystemeDeCharges) -> sp.Expr:
    """`V_2 = (1 / 4 pi eps_0 r^3) somme q_i [3/2 (r_i . u_r)^2 - r_i^2 / 2]`."""
    termes = (
        sp.sympify(q)
        * (
            sp.Rational(3, 2) * (_scalaire(point) / DISTANCE) ** 2
            - sum(c**2 for c in point) / sp.Integer(2)
        )
        for q, point in systeme.charges
    )
    return sp.simplify(COULOMB * sp.Add(*termes) / DISTANCE**3)


def champ_du_dipole(moment: Point) -> Vecteur:
    """`E = [3 (p . r) r / r^5 - p / r^3] / (4 pi eps_0)`."""
    projection = _scalaire(moment)
    x, y, z = (
        COULOMB * (3 * projection * axe / DISTANCE**5 - c / DISTANCE**3)
        for c, axe in zip(moment, CARTESIENNES, strict=True)
    )
    return (x, y, z)
