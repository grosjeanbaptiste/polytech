"""Le transformateur, parfait puis reel.

Le syllabus d'electricite generale (16.26). Deux enroulements de `N_1`
et `N_2` spires sur un circuit magnetique parfait : un meme flux les
traverse, donc `u_2 = (N_2 / N_1) u_1` (16.26.2) ; la reluctance etant
negligeable, `N_1 i_1 + N_2 i_2 = 0`, donc `i_2 = -(N_1 / N_2) i_1`
(16.26.4). La puissance se conserve (16.26.5), et le rapport de
transformation `N_2 / N_1` change la grandeur d'une tension, d'un courant
ou d'une impedance (16.26.6).

Le transformateur reel equivaut a un transformateur parfait dont chaque
enroulement porte en serie sa resistance et son inductance propre de
dispersion : `u_1 = r_1 i_1 + l_1 di_1/dt + N_1 dPhi/dt` (16.26.8).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


@dataclass(frozen=True, slots=True)
class TransformateurParfait:
    """Les nombres de spires du primaire et du secondaire."""

    primaire: int
    secondaire: int

    def __post_init__(self) -> None:
        """Refuse un enroulement sans spire."""
        for nombre in (self.primaire, self.secondaire):
            if nombre <= 0:
                msg = f"nombre de spires nul ou negatif : {nombre}"
                raise ValueError(msg)

    def rapport(self) -> sp.Rational:
        """Le rapport de transformation `N_2 / N_1`."""
        return sp.Rational(self.secondaire, self.primaire)

    def tension_secondaire(self, tension: Grandeur) -> sp.Expr:
        """`u_2 = (N_2 / N_1) u_1` (16.26.2)."""
        return self.rapport() * tension

    def courant_secondaire(self, courant: Grandeur) -> sp.Expr:
        """`i_2 = -(N_1 / N_2) i_1` (16.26.4)."""
        return -courant / self.rapport()

    def impedance_vue_du_primaire(self, charge: Grandeur) -> sp.Expr:
        """`u_1 / i_1 = (N_1 / N_2)^2 Z` pour une charge `Z` au secondaire (16.26.6)."""
        return charge / self.rapport() ** 2


def tension_d_un_enroulement(
    resistance: Grandeur,
    dispersion: Grandeur,
    spires: int,
    courant: sp.Expr,
    flux: sp.Expr,
) -> sp.Expr:
    """`u = r i + l di/dt + N dPhi/dt` : (16.26.8) et (16.26.9)."""
    induite = spires * sp.diff(flux, TEMPS)
    return resistance * courant + dispersion * sp.diff(courant, TEMPS) + induite
