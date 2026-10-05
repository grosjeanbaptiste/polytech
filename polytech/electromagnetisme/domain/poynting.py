"""Le theoreme de Poynting.

Le syllabus d'electricite generale (14.14 et 14.16). L'energie du champ
d'un systeme croit, par unite de temps et de volume, de
`E . dD/dt + H . dB/dt` (14.14.1). Pour la partie du champ qu'entoure une
surface fermee, cet accroissement est la puissance des sources, moins
l'effet Joule, moins le flux sortant du vecteur `P = E ^ H` (14.16.2) :
tout se passe comme si l'energie s'ecoulait dans le champ.

Sous forme locale, `E . dD/dt + H . dB/dt + E . i + div(E ^ H) = 0` des
que les equations de Maxwell sont satisfaites.

Entre des conducteurs parfaits, le vecteur de Poynting leur est
parallele : l'energie voyage dans l'isolant, et les conducteurs la
guident. S'ils resistent, une composante normale entre par leur surface
et vaut l'effet Joule. Pour des charges et des aimants au repos, le
vecteur n'est pas nul mais son flux a travers une surface fermee l'est.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

if TYPE_CHECKING:
    Vecteur = tuple[Any, Any, Any]


def vecteur_de_poynting(champ: Vecteur, magnetique: Vecteur) -> Vecteur:
    """`P = E ^ H` (14.16.2), en watts par metre carre."""
    e_x, e_y, e_z = champ
    h_x, h_y, h_z = magnetique
    return (e_y * h_z - e_z * h_y, e_z * h_x - e_x * h_z, e_x * h_y - e_y * h_x)


def _scalaire(gauche: Vecteur, droite: Vecteur) -> sp.Expr:
    return sp.Add(*(a * b for a, b in zip(gauche, droite, strict=True)))


def _derivee(vecteur: Vecteur) -> Vecteur:
    x, y, z = (sp.diff(sp.sympify(composante), TEMPS) for composante in vecteur)
    return (x, y, z)


def puissance_emmagasinee(
    champ: Vecteur, magnetique: Vecteur, deplacement: Vecteur, induction: Vecteur
) -> sp.Expr:
    """`E . dD/dt + H . dB/dt` : l'integrande de (14.14.1), en W/m^3."""
    electrique = _scalaire(champ, _derivee(deplacement))
    return electrique + _scalaire(magnetique, _derivee(induction))


def residu_de_poynting(
    champ: Vecteur,
    magnetique: Vecteur,
    deplacement: Vecteur,
    induction: Vecteur,
    courant: Vecteur,
) -> sp.Expr:
    """`E . dD/dt + H . dB/dt + E . i + div(E ^ H)` : nul si Maxwell est satisfait."""
    emmagasinee = puissance_emmagasinee(champ, magnetique, deplacement, induction)
    x, y, z = (sp.sympify(c) for c in vecteur_de_poynting(champ, magnetique))
    flux = divergence_cartesienne((x, y, z))
    return sp.simplify(emmagasinee + _scalaire(champ, courant) + flux)
