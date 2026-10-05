"""Les forces electrostatiques : sur une charge, entre charges, sur un dipole.

Le syllabus d'electricite generale (2.4 et 2.20) :

* une charge placee dans un champ subit `F = q E` (2.4.1) ;
* le champ d'une charge ponctuelle donne la loi de Coulomb,
  `F = q q_1 R_1 / (4 pi eps_0 R_1^3)` (2.20.2), et la force d'un systeme
  est la somme vectorielle des forces de chaque charge (2.20.3) ;
* un doublet subit `F = q [E(r_1) - E(r_2)]` (2.20.4). Dans un champ
  uniforme la resultante est nulle et il reste le couple `p ^ E`
  (2.20.6) ; dans un champ non uniforme, `F = (p . grad) E` (2.20.8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.potentiel import (
    COULOMB,
    champ_du_potentiel,
    potentiel_des_charges,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import (
        Point,
        SystemeDeCharges,
    )
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur


def _triplet(matrice: sp.Matrix) -> Vecteur:
    x, y, z = matrice
    return (x, y, z)


def force_sur_une_charge(charge: Any, champ: Point) -> Vecteur:  # noqa: ANN401 -- exacte, reelle ou symbolique
    """`F = q E`."""
    return _triplet(sp.sympify(charge) * sp.Matrix(champ))


def force_de_coulomb(source: Any, depuis: Point, charge: Any, en: Point) -> Vecteur:  # noqa: ANN401 -- exactes, reelles ou symboliques
    """La force que la charge `source`, placee en `depuis`, exerce sur `charge`."""
    ecart = sp.Matrix(en) - sp.Matrix(depuis)
    distance = sp.sqrt(ecart.dot(ecart))
    if distance == 0:
        msg = "charges confondues : la force de Coulomb n'y est pas definie"
        raise ValueError(msg)
    return _triplet(
        COULOMB * sp.sympify(source) * sp.sympify(charge) * ecart / distance**3
    )


def force_du_systeme(systeme: SystemeDeCharges, charge: Any, en: Point) -> Vecteur:  # noqa: ANN401 -- exacte, reelle ou symbolique
    """`F = q E(r)`, `E` etant le champ de toutes les charges du systeme."""
    au_point = dict(zip(CARTESIENNES, en, strict=True))
    champ = champ_du_potentiel(potentiel_des_charges(systeme))
    x, y, z = (c.subs(au_point) for c in champ)
    return force_sur_une_charge(charge, (x, y, z))


def couple_sur_un_dipole(moment: Point, champ: Point) -> Vecteur:
    """`M = p ^ E` : nul quand le dipole est aligne sur le champ."""
    return _triplet(sp.Matrix(moment).cross(sp.Matrix(champ)))


def force_sur_un_dipole(moment: Point, champ: Point) -> Vecteur:
    """`F = (p . grad) E` : nulle dans un champ uniforme."""
    x, y, z = (
        sum(
            (
                p * sp.diff(sp.sympify(composante), axe)
                for p, axe in zip(moment, CARTESIENNES, strict=True)
            ),
            sp.Integer(0),
        )
        for composante in champ
    )
    return (x, y, z)
