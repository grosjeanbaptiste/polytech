"""Annexe du TP 7, section 4 : le filtre de Sallen-Key et ses quatre methodes.

La fonction de transfert de la page 7 est ecrite telle que l'annexe
l'imprime, puis confrontee a l'analyse nodale du montage ; les composants
des quatre methodes de conception sont reportes dans les formules de la
pulsation propre et du facteur de qualite.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electronique.domain.methodes_sallen_key import (
    Methode,
    cellule_normalisee,
    resistances_de_gain,
)
from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas

S: Final = sp.Symbol("s")
R1, R2, C1, C2, MU = sp.symbols("R_1 R_2 C_1 C_2 mu", positive=True)
W0, Q = sp.symbols("omega_0 Q", positive=True)


def _cellule() -> SallenKeyPasseBas:
    return SallenKeyPasseBas(R1, R2, C1, C2, MU)


def transfert_de_l_annexe() -> sp.Expr:
    """`H(s)` de la page 7, recopiee."""
    produit = R1 * R2 * C1 * C2
    amortissement = 1 / (R1 * C1) + 1 / (R2 * C1) + (1 - MU) / (R2 * C2)
    return (MU / produit) / (S**2 + amortissement * S + 1 / produit)


def transfert_par_l_analyse_nodale() -> sp.Expr:
    """La meme, tiree du montage par la bibliotheque."""
    return _cellule().transfert(S)


def pulsation_et_qualite() -> tuple[sp.Expr, sp.Expr]:
    """`w0` et `Q` de la page 7, en fonction des composants."""
    cellule = _cellule()
    return cellule.pulsation(), cellule.qualite()


def composants(methode: Methode) -> SallenKeyPasseBas:
    """Les composants normalises que la methode donne pour `w0` et `Q`."""
    return cellule_normalisee(methode, Q, W0)


def pulsation_et_qualite_de_la_methode(methode: Methode) -> tuple[sp.Expr, sp.Expr]:
    """Ce que valent `w0` et `Q` une fois ces composants reportes."""
    cellule = composants(methode)
    return sp.simplify(cellule.pulsation()), sp.simplify(cellule.qualite())


def pont_de_gain(*, gain: float, r1: float, r2: float) -> tuple[float, float]:
    """`(Ra, Rb)` : le gain voulu, et `Ra // Rb = R1 + R2` (page 9)."""
    return resistances_de_gain(gain, r1 + r2)
