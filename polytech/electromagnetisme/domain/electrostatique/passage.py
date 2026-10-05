"""Les relations de passage a la traversee d'une surface chargee.

Le syllabus d'electricite generale (2.17 et 2.18), `u_n1` etant la normale
dirigee du cote de la face 1 et `u_n2 = -u_n1` :

* la composante tangentielle du champ est continue,
  `u_n1 ^ E_1 + u_n2 ^ E_2 = 0` (2.17.4) ;
* la composante normale du deplacement saute de la charge surfacique,
  `u_n1 . D_1 + u_n2 . D_2 = sigma` (2.18.2) ; pour le champ, le saut vaut
  `(sigma + sigma') / eps_0` (2.18.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import Point
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur


def _difference(cote_1: Point, cote_2: Point) -> sp.Matrix:
    return sp.Matrix(cote_1) - sp.Matrix(cote_2)


def saut_normal(cote_1: Point, cote_2: Point, *, normale_1: Point) -> sp.Expr:
    """`u_n1 . A_1 + u_n2 . A_2` : le saut de la composante normale."""
    return sp.simplify(sp.Matrix(normale_1).dot(_difference(cote_1, cote_2)))


def residu_tangentiel(cote_1: Point, cote_2: Point, *, normale_1: Point) -> Vecteur:
    """`u_n1 ^ A_1 + u_n2 ^ A_2` : nul si la composante tangentielle est continue."""
    x, y, z = sp.Matrix(normale_1).cross(_difference(cote_1, cote_2))
    return (sp.simplify(x), sp.simplify(y), sp.simplify(z))
