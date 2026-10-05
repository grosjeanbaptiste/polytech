"""Le vecteur champ magnetique, le theoreme d'Ampere et le potentiel scalaire.

Le syllabus d'electricite generale (8.16 et 8.20) : la circulation de `B`
enlace les courants reels **et** fictifs (8.16.1). En posant

    H = B / mu_0 - M                                               (8.16.2)

on elimine les seconds : `circulation de H = I` (8.16.3), soit `rot H = i`
(8.16.4) -- c'est le theoreme d'Ampere. Pour un conducteur cylindrique de
rayon `a` parcouru par `I` (exemple 8.16.1),

    H_phi = I rho / (2 pi a^2) dedans,    H_phi = I / (2 pi rho) dehors.

Hors des courants, `rot H = 0` : `H = -grad U` (8.20.1), `U` etant le
potentiel scalaire magnetique. Il est multiforme -- un tour autour d'un
courant `I` le fait baisser de `I` (8.20.2) -- et vaut, pour un circuit,
`U = I Omega / (4 pi)`, `Omega` etant l'angle solide sous lequel on voit le
circuit (8.20.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.double_couche import (
    angle_solide_d_un_disque,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, ORDONNEE, RAYON

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import Point
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur


def champ_magnetique(induction: Point, aimante: Point) -> Vecteur:
    """`H = B / mu_0 - M`."""
    x, y, z = (
        sp.sympify(b) / PERMEABILITE - sp.sympify(m)
        for b, m in zip(induction, aimante, strict=True)
    )
    return (x, y, z)


def champ_d_un_conducteur_cylindrique(courant: object, rayon: object) -> sp.Expr:
    """`H_phi(rho)` : lineaire dans le conducteur, en `1 / rho` au-dehors."""
    intensite, a = sp.sympify(courant), sp.sympify(rayon)
    dedans = intensite * RAYON / (2 * sp.pi * a**2)
    dehors = intensite / (2 * sp.pi * RAYON)
    return sp.Piecewise((dedans, a > RAYON), (dehors, True))


def potentiel_scalaire_du_fil(courant: object) -> sp.Expr:
    """`U = -I phi / (2 pi)`, `phi` etant l'azimut autour du fil porte par l'axe z."""
    return -sp.sympify(courant) * sp.atan2(ORDONNEE, ABSCISSE) / (2 * sp.pi)


def potentiel_scalaire_d_une_boucle(
    courant: object, rayon: object, cote: object
) -> sp.Expr:
    """`U = I Omega / (4 pi)` sur l'axe d'une boucle : l'angle solide de son disque."""
    return sp.sympify(courant) * angle_solide_d_un_disque(rayon, cote) / (4 * sp.pi)
