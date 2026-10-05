"""L'amplificateur a un pole et a retour capacitif : la compensation de Miller.

Syllabus, section 3.8.3. Un amplificateur inverseur de gain `-A(s)`,
`A(s) = A_DC / (1 + s / w_c)`, attaque a travers `R_S` et reboucle par
`C_f`. Par le theoreme de Miller, avec `beta = C_f R_S` :

    V_2 / V_i = -A(s) / (1 + s beta (1 + A(s)))
              = -A_DC / ((beta / w_c) s^2 + s (beta (1 + A_DC) + 1 / w_c) + 1).

Ses deux poles sont reels et se **separent** : le plus lent est celui de
la capacite de Miller, `-1 / (beta (1 + A_DC))` ; l'autre, par les
relations de Viete, vaut `-w_c (1 + A_DC)`. Sous le dixieme de `w_c`,
l'amplificateur se comporte comme s'il etait parfait : c'est
l'approximation de Miller.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp


def transfert_a_retour_capacitif(
    variable: sp.Expr, *, continu: sp.Expr, coupure: sp.Expr, constante: sp.Expr
) -> sp.Expr:
    """`-A_DC / ((beta / w_c) s^2 + s (beta (1 + A_DC) + 1 / w_c) + 1)`."""
    carre = constante / coupure * variable**2
    lineaire = (constante * (1 + continu) + 1 / coupure) * variable
    return -continu / (carre + lineaire + 1)


def _exiger_des_grandeurs(*grandeurs: float) -> None:
    if any(grandeur <= 0 for grandeur in grandeurs):
        msg = f"grandeur nulle ou negative : {grandeurs}"
        raise ValueError(msg)


def poles_a_retour_capacitif(
    *, continu: float, coupure: float, constante: float
) -> tuple[float, float]:
    """Les deux poles exacts, le plus lent d'abord."""
    _exiger_des_grandeurs(continu, coupure, constante)
    carre = constante / coupure
    lineaire = constante * (1 + continu) + 1 / coupure
    racine = math.sqrt(lineaire**2 - 4 * carre)
    rapide = -(lineaire + racine) / (2 * carre)
    return 1 / (carre * rapide), rapide


def poles_separes(
    *, continu: float, coupure: float, constante: float
) -> tuple[float, float]:
    """`(-1 / (beta (1 + A_DC)), -w_c (1 + A_DC))` : l'approximation de Miller."""
    _exiger_des_grandeurs(continu, coupure, constante)
    return -1 / (constante * (1 + continu)), -coupure * (1 + continu)
