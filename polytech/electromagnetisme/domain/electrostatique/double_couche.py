"""La double couche electrique, et le saut de potentiel qu'elle impose.

Le syllabus d'electricite generale (2.16) : le potentiel est continu a la
traversee d'une surface (2.16.1), sauf si elle porte une double couche de
moment surfacique `P_S`, normal a la surface :

    V_1 - V_2 = P_S / eps_0                           (2.16.2), (2.16.5)
    V(P) = P_S Omega(P) / (4 pi eps_0)                (2.16.4)

ou `Omega` est l'angle solide sous lequel on voit la face 1 -- compte
negativement si c'est la face 2 que l'on voit. Pour une couche fermee,
`Omega` vaut 0 dehors, `4 pi` dedans et `2 pi` sur la surface (2.16.6).
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB


def potentiel_de_la_double_couche(moment: object, angle_solide: object) -> sp.Expr:
    """`V = P_S Omega / (4 pi eps_0)`, pour une double couche uniforme."""
    return COULOMB * sp.sympify(moment) * sp.sympify(angle_solide)


def saut_de_potentiel(moment: object) -> sp.Expr:
    """`V_1 - V_2 = P_S / eps_0`, de la face 1 a la face 2."""
    return sp.sympify(moment) / PERMITTIVITE


def angle_solide_d_un_disque(rayon: object, distance: object) -> sp.Expr:
    """L'angle solide d'un disque vu de son axe, du signe du cote ou l'on est."""
    a, d = sp.sympify(rayon), sp.sympify(distance)
    if d == 0:
        msg = "point sur la couche : l'angle solide n'y est pas defini"
        raise ValueError(msg)
    return sp.sign(d) * 2 * sp.pi * (1 - sp.Abs(d) / sp.sqrt(d**2 + a**2))
