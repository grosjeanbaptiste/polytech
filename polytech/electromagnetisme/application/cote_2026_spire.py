"""Exercice cote 2026 (Charleroi), question 2 : deux fils et une petite spire.

Le conducteur 1 est en `(0, a)`, son courant sort de la feuille ; le
conducteur 2 en `(0, -a)`, son courant y entre. La spire `dS` est en
`P = (R, 0)`, sa normale selon `x`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
    integrale_de_biot_et_savart,
)
from polytech.electromagnetisme.domain.magnetostatique.superposition import Fil

MU0: Final = PERMEABILITE
I, IP, A, R, DS = sp.symbols("I I' a R dS", positive=True)  # noqa: E741 -- le courant de l'enonce
_PORTEE: Final = 2e4
"""La demi-longueur, en metres, du fil qui tient lieu de fil illimite."""


def _champ_d_un_fil(courant: sp.Expr, ordonnee: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`H` en `P`, le courant compte positif quand il sort de la feuille."""
    r_x, r_y = R, -ordonnee
    distance = sp.sqrt(r_x**2 + r_y**2)
    module = Fil(courant, distance).module_du_champ()
    return -module * r_y / distance, module * r_x / distance


def champ() -> tuple[sp.Expr, sp.Expr]:
    """`(H_x, H_y)` en `P` : les composantes selon `y` se compensent."""
    du_premier = _champ_d_un_fil(I, A)
    du_second = _champ_d_un_fil(-I, -A)
    return (
        sp.simplify(du_premier[0] + du_second[0]),
        sp.simplify(du_premier[1] + du_second[1]),
    )


def mutuelle() -> sp.Expr:
    """`L_M = phi_M / I`, le flux a travers `dS` par ampere des deux fils."""
    h_x, _ = champ()
    return sp.simplify(MU0 * h_x * DS / I)


def force() -> sp.Expr:
    """`F_x = I I' dL_M/dR`, a courants constants."""
    return sp.simplify(I * IP * mutuelle().diff(R))


def distance_de_la_force_maximale() -> sp.Expr:
    """Le `R` ou la force est la plus grande en module."""
    (distance,) = sp.solve(force().diff(R), R)
    return distance


def champ_par_biot_et_savart(demi_ecart: float, distance: float) -> tuple[float, float]:
    """`(H_x, H_y)` pour un ampere, les deux fils integres element par element."""
    z = sp.Symbol("z", real=True)
    point = (sp.Float(distance), sp.Integer(0), sp.Integer(0))
    total = [0.0, 0.0]
    for courant, ordonnee in ((1.0, demi_ecart), (-1.0, -demi_ecart)):
        fil = (sp.Integer(0), sp.Float(ordonnee), z)
        en_x, en_y, _ = integrale_de_biot_et_savart(fil, z, -_PORTEE, _PORTEE, point)
        total[0] += courant * en_x / (4 * sp.pi)
        total[1] += courant * en_y / (4 * sp.pi)
    return float(total[0]), float(total[1])
