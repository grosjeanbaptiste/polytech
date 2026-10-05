"""La formule de Biot et Savart : l'induction d'un element de courant.

Le syllabus d'electricite generale (8.8 et 8.9) : le potentiel vecteur est
newtonien, `A = (mu_0 / 4 pi) somme q_i v_i / R_i` (8.8.1), d'ou

    B = (mu_0 / 4 pi) somme q_i v_i ^ R_i / R_i^3                  (8.8.6)

et, pour un circuit parcouru par un courant `I`,

    dB = (mu_0 / 4 pi) dI dl ^ R / R^3                             (8.9.1)

`R` allant de l'element de courant au point ou l'on calcule. Le sens de
`dB` est celui de la regle du tire-bouchon. `dB` n'est qu'un intermediaire
de calcul : seule son integrale sur un circuit ferme a un sens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import mpmath
import sympy as sp

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import Point
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur

PERMEABILITE: Final = sp.Symbol("mu_0", positive=True)
"""`mu_0`, la permeabilite du vide, gardee symbolique."""

_TRONCONS: Final = 32
"""Le circuit est integre par morceaux : un point proche du fil le demande."""


def _noyau(vers_le_point: Point) -> sp.Matrix:
    ecart = sp.Matrix(vers_le_point)
    distance = sp.sqrt(ecart.dot(ecart))
    if distance == 0:
        msg = "point confondu avec la source : R / R^3 n'y est pas defini"
        raise ValueError(msg)
    return ecart / distance**3


def _triplet(matrice: sp.Matrix) -> Vecteur:
    x, y, z = matrice
    return (x, y, z)


def induction_d_un_element(
    courant: object, element: Point, vers_le_point: Point
) -> Vecteur:
    """`dB = (mu_0 / 4 pi) I dl ^ R / R^3`."""
    produit = sp.Matrix(element).cross(_noyau(vers_le_point))
    return _triplet(PERMEABILITE * sp.sympify(courant) * produit / (4 * sp.pi))


def induction_d_une_charge(
    charge: object, vitesse: Point, vers_le_point: Point
) -> Vecteur:
    """`(mu_0 / 4 pi) q v ^ R / R^3` : un terme de la somme (8.8.6)."""
    return induction_d_un_element(charge, vitesse, vers_le_point)


def integrale_de_biot_et_savart(
    courbe: Vecteur, parametre: sp.Symbol, debut: float, fin: float, point: Point
) -> tuple[float, float, float]:
    """`integrale de dl ^ R / R^3` sur le circuit ; `B` en vaut `mu_0 I / 4 pi` fois."""
    source = sp.Matrix(courbe)
    ecart = sp.Matrix(point) - source
    integrande = sp.diff(source, parametre).cross(ecart) / ecart.dot(
        ecart
    ) ** sp.Rational(3, 2)
    bornes = [debut + (fin - debut) * rang / _TRONCONS for rang in range(_TRONCONS + 1)]
    x, y, z = (
        float(mpmath.quad(sp.lambdify(parametre, composante, "mpmath"), bornes))
        for composante in integrande
    )
    return (x, y, z)
