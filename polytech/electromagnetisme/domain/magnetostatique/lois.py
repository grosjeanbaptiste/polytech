"""Les deux proprietes fondamentales du champ magnetique.

Symetriquement a l'electrostatique, les planches les nomment par le flux
et par la circulation -- mais les roles sont **echanges**, et c'est tout
l'interet du parallele que les planches tracent explicitement :

* le flux de `B` est **toujours nul** (`div B = 0`) : il n'y a pas de
  charge magnetique ;
* la circulation de `H` **ne l'est pas** : elle vaut le courant enlace,
  c'est le theoreme d'Ampere.

C'est l'inverse exact de l'electrostatique, ou la divergence vaut la
charge et la circulation est nulle. Confondre les deux est l'erreur que ce
parallele sert a prevenir.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
)

if TYPE_CHECKING:
    import sympy as sp


def est_un_champ_magnetique(champ: tuple[sp.Expr, sp.Expr, sp.Expr]) -> bool:
    """Vrai si `div B = 0` : condition necessaire, et jamais facultative."""
    return bool(divergence_cartesienne(champ) == 0)


def induction_du_fil_infini(courant: float, distance: float) -> float:
    """`B = mu_0 I / (2 pi r)` : Biot et Savart integre sur un fil droit."""
    if distance <= 0:
        msg = f"distance au fil nulle ou negative : {distance} m"
        raise ValueError(msg)
    return PERMEABILITE_DU_VIDE * courant / (2 * math.pi * distance)


def courant_enlace(champ: float, longueur: float) -> float:
    """`N I = H l` : le theoreme d'Ampere, lu dans le sens du probleme.

    C'est exactement le geste de l'exercice 6 sur les circuits
    magnetiques : on lit `H` sur la courbe du materiau, on multiplie par
    la longueur du contour, et l'on obtient les amperes-tours.
    """
    return champ * longueur


def flux_magnetique(induction: float, section: float) -> float:
    """`Phi = B S`, pour une induction uniforme et normale a la section."""
    if section <= 0:
        msg = f"section nulle ou negative : {section} m^2"
        raise ValueError(msg)
    return induction * section


def champ_depuis_l_induction(induction: float, relative: float = 1.0) -> float:
    """`H = B / (mu_r mu_0)` ; leve pour une permeabilite relative sous un."""
    if relative < 1:
        msg = (
            f"permeabilite relative sous l'unite : {relative} -- un materiau "
            f"ferromagnetique la porte a plusieurs milliers"
        )
        raise ValueError(msg)
    return induction / (relative * PERMEABILITE_DU_VIDE)
