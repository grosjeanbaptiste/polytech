"""Les courants electriques : conduction, convection, polarisation.

Le syllabus d'electricite generale (chapitre 7) range les charges en
mouvement par especes de corpuscules, `a = 0` designant ceux qui sont au
repos par rapport au conducteur. Chaque espece a sa charge volumique
dynamique `eta_a = N_a Q_a / Delta V` (7.3.1) et sa vitesse d'ensemble
`v_a` ; la charge statique est leur somme, `eta = somme eta_a` (7.3.5).

Le courant volumique `i = somme(q_i v_i) / Delta V` (7.4.1) se decompose en

    conduction    i_cd = somme sur a >= 1 des eta_a (v_a - v)      (7.4.3)
    convection    i_cv = eta v                                     (7.4.4)
    polarisation  i_p  = dP/dt                                     (7.4.5)

`v` etant la vitesse d'entrainement du corps. L'intensite a travers une
surface est le flux du courant volumique (7.5.1), et la conservation de
l'electricite s'ecrit `div(i_cd + i_cv) + d eta/dt = 0` (7.6.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.electromagnetisme.domain.electrostatique.moments import Point
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur


@dataclass(frozen=True, slots=True)
class Espece:
    """Une espece de corpuscules : sa charge volumique dynamique et sa vitesse."""

    charge_volumique: Any
    vitesse: Point


def _triplet(matrice: sp.Matrix) -> Vecteur:
    x, y, z = matrice
    return (x, y, z)


def _exiger_un_volume(volume: object) -> sp.Expr:
    etendue = sp.sympify(volume)
    if etendue.is_positive is not True:
        msg = f"volume strictement positif attendu, et non {etendue}"
        raise ValueError(msg)
    return etendue


def charge_dynamique(*, nombre: object, charge: object, volume: object) -> sp.Expr:
    """`eta_a = N_a Q_a / Delta V`."""
    return sp.sympify(nombre) * sp.sympify(charge) / _exiger_un_volume(volume)


def charge_statique(especes: Sequence[Espece]) -> sp.Expr:
    """`eta = somme des eta_a` : ce que voit l'electrostatique."""
    return sp.Add(*(sp.sympify(espece.charge_volumique) for espece in especes))


def courant_volumique(
    particules: Sequence[tuple[Any, Point]], *, volume: object
) -> Vecteur:
    """`i = somme(q_i v_i) / Delta V`, sur les particules de l'element de volume."""
    somme = sum((sp.sympify(q) * sp.Matrix(v) for q, v in particules), sp.zeros(3, 1))
    return _triplet(somme / _exiger_un_volume(volume))


def courant_de_conduction(especes: Sequence[Espece], entrainement: Point) -> Vecteur:
    """`i_cd = somme des eta_a (v_a - v)` : le mouvement relatif au conducteur."""
    corps = sp.Matrix(entrainement)
    termes = (
        sp.sympify(espece.charge_volumique) * (sp.Matrix(espece.vitesse) - corps)
        for espece in especes
    )
    return _triplet(sum(termes, sp.zeros(3, 1)))


def courant_de_convection(charge: object, entrainement: Point) -> Vecteur:
    """`i_cv = eta v` : la charge statique entrainee par le corps."""
    return _triplet(sp.sympify(charge) * sp.Matrix(entrainement))


def courant_de_polarisation(polarise: Point) -> Vecteur:
    """`i_p = dP/dt`."""
    x, y, z = (sp.diff(sp.sympify(composante), TEMPS) for composante in polarise)
    return (x, y, z)


def intensite_a_travers(courant: Point, normale: Point, aire: object) -> sp.Expr:
    """`I = u_n . i S`, pour un courant uniforme sur une surface plane."""
    unitaire = sp.Matrix(normale)
    if sp.simplify(unitaire.dot(unitaire) - 1) != 0:
        msg = f"normale unitaire attendue, et non {normale}"
        raise ValueError(msg)
    return sp.simplify(unitaire.dot(sp.Matrix(courant)) * sp.sympify(aire))
