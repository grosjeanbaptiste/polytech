"""Les moyennes macroscopiques : d'un nuage de charges a une densite continue.

Le syllabus d'electricite generale (1.5) remplace « la distribution reelle
discrete des charges individuelles par une distribution continue fictive
equivalente ». La moyenne se prend sur un element assez grand pour contenir
beaucoup de charges, assez petit pour ne pas gommer leur repartition :

* charge volumique `eta = somme(q_i) / Delta V` (1.5.1) ;
* charge surfacique `sigma = somme(q_i) / Delta S` (1.5.5), et `sigma =
  eta a` pour une couche d'epaisseur `a` (1.5.4) ;
* charge lineique `lambda = somme(q_i) / Delta l` (1.5.8), et `lambda =
  eta s` pour un filet de section `s` (1.5.7).

Des charges ponctuelles s'ecrivent elles-memes comme une charge volumique,
`eta(r) = somme q_i delta(r - r_i)` (1.6.8).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.electromagnetisme.domain.symboles import CARTESIENNES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Reel = Fraction | float
"""Une position ou une charge : un reel, ou une fraction quand le compte est exact."""


def _exiger_une_etendue(etendue: Reel) -> None:
    if etendue <= 0:
        msg = f"etendue strictement positive attendue, et non {etendue}"
        raise ValueError(msg)


def charge_moyenne(charges: Sequence[Reel], etendue: Reel) -> Reel:
    """La somme des charges de l'element, rapportee a son etendue."""
    _exiger_une_etendue(etendue)
    return sum(charges) / etendue


def charges_de_la_fenetre(
    positions: Sequence[Reel], debut: Reel, largeur: Reel
) -> list[Reel]:
    """Les positions comprises dans `[debut, debut + largeur[`."""
    _exiger_une_etendue(largeur)
    return [x for x in positions if debut <= x < debut + largeur]


def charge_surfacique_d_une_couche(volumique: float, epaisseur: float) -> float:
    """`sigma = eta a`, pour une couche d'epaisseur `a` tres faible."""
    return volumique * epaisseur


def charge_lineique_d_un_filet(volumique: float, section: float) -> float:
    """`lambda = eta s`, pour un filet de section `s` tres petite."""
    return volumique * section


def charge_volumique_d_une_sphere(charge: float, rayon: float) -> float:
    """`eta = Q / (4 pi R^3 / 3)`, pour une sphere uniformement chargee."""
    if rayon <= 0:
        msg = f"rayon strictement positif attendu, et non {rayon}"
        raise ValueError(msg)
    return charge / (4 * math.pi * rayon**3 / 3)


def _dirac_en(point: tuple[Any, Any, Any]) -> sp.Expr:
    axes = zip(CARTESIENNES, point, strict=True)
    return sp.Mul(*(sp.DiracDelta(axe - coordonnee) for axe, coordonnee in axes))


def charge_volumique_des_charges_ponctuelles(
    charges: Mapping[tuple[Any, Any, Any], Any],
) -> sp.Expr:
    """`eta(r) = somme q_i delta(x - x_i) delta(y - y_i) delta(z - z_i)`."""
    pics = (q * _dirac_en(point) for point, q in charges.items())
    return sp.Add(*pics)
