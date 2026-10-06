"""Exercice cote 2024 (Charleroi), question 2 : la force sur un fil entre deux autres.

Le conducteur 1 est a l'origine, son courant sort de la feuille ; le
conducteur 2 est en `(d, 0)`, son courant y entre. Le fil d'epreuve
perce la feuille en `P = (0, y)`.
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
I1, I2, D = sp.symbols("I_1 I_2 d", positive=True)
Y: Final = sp.Symbol("y", real=True, nonzero=True)
COURANT: Final = sp.Symbol("I", real=True)
VALEURS: Final = {I1: 1, I2: 2, D: 1}
RAYON: Final = 2e-3
"""Le rayon des deux conducteurs, en metres."""
_PORTEE: Final = 2e4
"""La demi-longueur, en metres, du fil qui tient lieu de fil illimite."""


def _champ_d_un_fil(courant: sp.Expr, abscisse: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`H` en `P`, pour un courant compte positif quand il sort de la feuille.

    Le champ tourne autour du fil : sa direction est `u_z ^ u_r`, soit
    `(-r_y, r_x)/r` pour un rayon `(r_x, r_y)` allant du fil au point.
    """
    r_x, r_y = -abscisse, Y
    distance = sp.sqrt(r_x**2 + r_y**2)
    module = Fil(courant, distance).module_du_champ()
    return -module * r_y / distance, module * r_x / distance


def champ() -> tuple[sp.Expr, sp.Expr]:
    """`(H_x, H_y)` en `P`, somme des champs des deux conducteurs."""
    du_premier = _champ_d_un_fil(I1, sp.Integer(0))
    du_second = _champ_d_un_fil(-I2, D)
    return (
        sp.simplify(du_premier[0] + du_second[0]),
        sp.simplify(du_premier[1] + du_second[1]),
    )


def force_lineique(ordonnee: object) -> tuple[sp.Expr, sp.Expr]:
    """`dF/dz = mu0 I u_z ^ H`, en N/m, avec les valeurs de l'enonce."""
    h_x, h_y = (c.subs(VALEURS).subs(Y, ordonnee) for c in champ())
    return sp.simplify(-MU0 * COURANT * h_y), sp.simplify(MU0 * COURANT * h_x)


def ordonnees() -> tuple[sp.Expr, ...]:
    """Les `y` ou la force n'a pas de composante selon `y` : `H_x = 0`."""
    h_x, _ = champ()
    return tuple(sorted(sp.solve(sp.Eq(h_x.subs(VALEURS), 0), Y)))


def champ_chiffre(ordonnee: float) -> tuple[float, float]:
    """`(H_x, H_y)` en A/m, avec les valeurs de l'enonce."""
    h_x, h_y = (c.subs(VALEURS).subs(Y, ordonnee) for c in champ())
    return float(h_x), float(h_y)


def champ_par_biot_et_savart(ordonnee: float) -> tuple[float, float]:
    """Le meme champ, les deux fils integres element par element."""
    z = sp.Symbol("z", real=True)
    point = (sp.Integer(0), sp.Float(ordonnee), sp.Integer(0))
    total = [0.0, 0.0]
    for courant, abscisse in ((1.0, 0), (-2.0, 1)):
        fil = (sp.Integer(abscisse), sp.Integer(0), z)
        en_x, en_y, _ = integrale_de_biot_et_savart(fil, z, -_PORTEE, _PORTEE, point)
        total[0] += courant * en_x / (4 * sp.pi)
        total[1] += courant * en_y / (4 * sp.pi)
    return float(total[0]), float(total[1])
