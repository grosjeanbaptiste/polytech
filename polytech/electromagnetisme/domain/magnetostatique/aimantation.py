"""Le moment magnetique, l'aimantation et les courants fictifs.

Le syllabus d'electricite generale (8.10, 8.11 et 8.15), miroir magnetique
des paragraphes 2.3, 2.8 et 2.9 :

* une charge liee qui parcourt son orbite en un temps `T` a le moment
  magnetique amperien `m = (q / T) S u_n` (8.10.14) ; un circuit filiforme
  plan parcouru par `I`, `m = (I / 2) circulation de r ^ dl = I S u_n`
  (8.15.12), independant de l'origine ;
* l'aimantation est le moment magnetique volumique, `M = somme(m_i) / Delta V`
  (8.10.16) ;
* le potentiel vecteur d'un moment vaut `A = (mu_0 / 4 pi) m ^ r / r^3`
  (8.10.21), « analogue de la relation (2.7.8) des dipoles » ;
* l'aimantation equivaut a des courants **fictifs**, `i' = rot M` en volume
  (8.11.1a) et `i'_S = u_n ^ M` en surface, `u_n` etant la normale orientee
  vers l'interieur (8.11.1b). Leur total est nul (8.11.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    rotationnel_cartesien,
)
from polytech.electromagnetisme.domain.electrostatique.polarisation import polarisation
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.electromagnetisme.domain.electrostatique.moments import Point
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur

_POSITION = sp.Matrix(CARTESIENNES)
_DISTANCE = sp.sqrt(_POSITION.dot(_POSITION))


def _triplet(matrice: sp.Matrix) -> Vecteur:
    x, y, z = matrice
    return (x, y, z)


def moment_d_un_circuit_plan(courant: object, aire: object, normale: Point) -> Vecteur:
    """`m = I S u_n`, `u_n` etant associee au sens de parcours du circuit."""
    return _triplet(sp.sympify(courant) * sp.sympify(aire) * sp.Matrix(normale))


def moment_amperien(
    *, charge: object, periode: object, aire: object, normale: Point
) -> Vecteur:
    """`m = (q / T) S u_n` : le moment d'une charge liee sur son orbite."""
    duree = sp.sympify(periode)
    if duree.is_positive is not True:
        msg = f"periode strictement positive attendue, et non {duree}"
        raise ValueError(msg)
    return moment_d_un_circuit_plan(sp.sympify(charge) / duree, aire, normale)


def moment_d_un_circuit(
    courant: object, courbe: Vecteur, parametre: sp.Symbol, debut: object, fin: object
) -> Vecteur:
    """`m = (I / 2) circulation de r ^ dl`, sur le contour parametre."""
    position = sp.Matrix(courbe)
    integrande = position.cross(sp.diff(position, parametre))
    integrale = integrande.applyfunc(lambda c: sp.integrate(c, (parametre, debut, fin)))
    return _triplet(sp.simplify(sp.sympify(courant) * integrale / 2))


def aimantation(moments: Sequence[Point], volume: object) -> Vecteur:
    """`M = somme(m_i) / Delta V` : le meme quotient que la polarisation."""
    return polarisation(moments, volume)


def potentiel_vecteur_du_moment(moment: Point) -> Vecteur:
    """`A = (mu_0 / 4 pi) m ^ r / r^3`, pour un moment place a l'origine."""
    return _triplet(
        PERMEABILITE * sp.Matrix(moment).cross(_POSITION) / (4 * sp.pi * _DISTANCE**3)
    )


def induction_du_moment(moment: Point) -> Vecteur:
    """`B = (mu_0 / 4 pi) [3 (m . r) r / r^5 - m / r^3]`."""
    vecteur = sp.Matrix(moment)
    crochet = (
        3 * vecteur.dot(_POSITION) * _POSITION / _DISTANCE**5 - vecteur / _DISTANCE**3
    )
    return _triplet(PERMEABILITE * crochet / (4 * sp.pi))


def courant_volumique_fictif(aimante: Point) -> Vecteur:
    """`i' = rot M`."""
    x, y, z = (sp.sympify(composante) for composante in aimante)
    return rotationnel_cartesien((x, y, z))


def courant_surfacique_fictif(aimante: Point, *, normale_interieure: Point) -> Vecteur:
    """`i'_S = u_n ^ M`, `u_n` pointant vers l'interieur du milieu aimante."""
    normale = sp.Matrix(normale_interieure)
    if sp.simplify(normale.dot(normale) - 1) != 0:
        msg = f"normale unitaire attendue, et non {normale_interieure}"
        raise ValueError(msg)
    return _triplet(normale.cross(sp.Matrix(aimante)))
