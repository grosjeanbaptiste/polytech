"""Exercice cote 2026 (Charleroi), question 3 : deux bobines couplees en quadrature.

Le primaire (`e`, `R1`, `L1`) induit dans le secondaire (`L2`, `C`,
`R2`). Seule la maille du secondaire sert : elle lie `i2` a `i1`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.bornes_pointees import (
    signe_de_la_mutuelle,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

T: Final = TEMPS
R2, L2, L12, C, W = sp.symbols("R_2 L_2 L_12 C omega", positive=True)
FREQUENCE: Final = 50
RESISTANCE: Final = 10
"""`R2`, en ohms."""
RAPPORT_VOULU: Final = sp.Rational(1, 2)


def signe() -> int:
    """Le signe de la mutuelle : `i1` entre sans point, `i2` par le point."""
    return signe_de_la_mutuelle(premier_par_le_point=False, second_par_le_point=True)


def rapport(*, signe_de_la_mutuelle: int | None = None) -> sp.Expr:
    """`I2/I1`, tire de `0 = (R2 + j w L2 + 1/(j w C)) I2 +- j w L12 I1`."""
    sens = signe() if signe_de_la_mutuelle is None else signe_de_la_mutuelle
    impedance = R2 + sp.I * W * L2 + 1 / (sp.I * W * C)
    return -sens * sp.I * W * L12 / impedance


def condition() -> sp.Eq:
    """Ce qui rend le rapport imaginaire pur : sa partie reelle s'annule."""
    partie_reelle = sp.simplify(sp.re(sp.expand_complex(rapport())))
    (inductance,) = sp.solve(partie_reelle, L2)
    return sp.Eq(L2 * C, sp.simplify(inductance * C))


def produit_a_cinquante_hertz() -> sp.Expr:
    """`L2 C`, en secondes carrees, a 50 Hz."""
    return condition().rhs.subs(W, 2 * sp.pi * FREQUENCE)


def rapport_a_la_resonance(*, signe: int | None = None) -> sp.Expr:
    """`I2/I1` quand `L2 C w^2 = 1`."""
    return sp.simplify(rapport(signe_de_la_mutuelle=signe).subs(C, 1 / (L2 * W**2)))


def mutuelle() -> sp.Expr:
    """`L12`, en henrys, pour que `I2/I1` vaille un demi en module."""
    module = sp.Abs(rapport_a_la_resonance()).subs(
        {W: 2 * sp.pi * FREQUENCE, R2: RESISTANCE}
    )
    (valeur,) = sp.solve(sp.Eq(module, RAPPORT_VOULU), L12)
    return valeur


def residu_de_l_equation_du_secondaire() -> sp.Expr:
    """`R2 i2' + L2 i2'' + i2/C - L12 i1''`, pour `i1 = cos(w t)` et l'`i2` annonce."""
    primaire = sp.cos(W * T)
    secondaire = W * L12 / R2 * sp.cos(W * T + sp.pi / 2)
    maille = R2 * secondaire.diff(T) + L2 * secondaire.diff(T, 2) + secondaire / C
    return (maille + signe() * L12 * primaire.diff(T, 2)).subs(C, 1 / (L2 * W**2))


def resistance_ramenee() -> sp.Expr:
    """`(w L12)^2 / R2`, en ohms : ce que le secondaire accorde ajoute au primaire."""
    pulsation = 2 * sp.pi * FREQUENCE
    return sp.simplify((pulsation * mutuelle()) ** 2 / RESISTANCE)
