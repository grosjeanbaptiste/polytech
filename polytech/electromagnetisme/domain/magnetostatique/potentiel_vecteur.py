"""Le potentiel vecteur magnetique, et la liberte de jauge.

Le syllabus d'electricite generale (8.6 et 8.8) : le flux de l'induction a
travers toute surface fermee est nul (8.6.1), donc `div B = 0` (8.6.2) et
`B = rot A` (8.6.3). Le flux a travers un contour est alors la circulation
de `A` (8.6.4) : il ne depend pas de la surface posee sur le contour.

`A` n'est defini qu'au gradient d'une fonction pres : `A' = A + grad(phi)`
donne la meme induction. Dans les etats stationnaires, on leve
l'indetermination par la jauge de Coulomb, `div A = 0` (8.8.2), qu'un
changement de jauge preserve si `phi` est harmonique.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
    rotationnel_cartesien,
)
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.potentiel import Vecteur


def induction_du_potentiel(potentiel: Vecteur) -> Vecteur:
    """`B = rot A`."""
    return rotationnel_cartesien(potentiel)


def changement_de_jauge(potentiel: Vecteur, fonction: sp.Expr) -> Vecteur:
    """`A' = A + grad(phi)` : un autre potentiel vecteur pour la meme induction."""
    x, y, z = (
        composante + sp.diff(fonction, axe)
        for composante, axe in zip(potentiel, CARTESIENNES, strict=True)
    )
    return (x, y, z)


def est_en_jauge_de_coulomb(potentiel: Vecteur) -> bool:
    """Vrai si `div A = 0`."""
    return bool(sp.simplify(divergence_cartesienne(potentiel)) == 0)
