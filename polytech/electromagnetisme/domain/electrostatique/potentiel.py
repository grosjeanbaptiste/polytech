"""La fonction potentielle, le champ qui en derive, et ses lignes.

Le syllabus d'electricite generale (2.5 et 2.6) :

* premiere propriete : le champ electrostatique est irrotationnel, donc
  `E = -grad V` (2.5.5) ;
* deuxieme propriete : le potentiel est newtonien,
  `V(r) = (1 / 4 pi eps_0) somme q_i / |r - r_i|` (2.6.1) ;
* une ligne de champ est tangente au champ en chacun de ses points,
  `dl ^ E = 0` (2.5.8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.symboles import CARTESIENNES

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import (
        SystemeDeCharges,
    )

Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]

COULOMB = 1 / (4 * sp.pi * PERMITTIVITE)
"""Le facteur `1 / (4 pi eps_0)` des equations « rationnelles »."""


def potentiel_des_charges(systeme: SystemeDeCharges) -> sp.Expr:
    """`V(r) = (1 / 4 pi eps_0) somme q_i / R_i`, au point courant `(x, y, z)`."""
    termes = (
        sp.sympify(q)
        / sp.sqrt(
            sum((axe - c) ** 2 for axe, c in zip(CARTESIENNES, point, strict=True))
        )
        for q, point in systeme.charges
    )
    return COULOMB * sp.Add(*termes)


def champ_du_potentiel(potentiel: sp.Expr) -> Vecteur:
    """`E = -grad V`."""
    x, y, z = (-sp.diff(potentiel, axe) for axe in CARTESIENNES)
    return (x, y, z)


def est_une_ligne_de_champ(
    champ: Vecteur, courbe: Vecteur, parametre: sp.Symbol
) -> bool:
    """Vrai si la courbe parametree est partout tangente au champ."""
    sur_la_courbe = dict(zip(CARTESIENNES, courbe, strict=True))
    tangente = sp.Matrix([sp.diff(c, parametre) for c in courbe])
    local = sp.Matrix(
        [sp.sympify(c).subs(sur_la_courbe, simultaneous=True) for c in champ]
    )
    return all(sp.simplify(c) == 0 for c in tangente.cross(local))
