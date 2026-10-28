"""La variation de masse volumique, rapportee a la masse volumique d'arret.

`regimes.variation_de_masse_volumique` rend `rho_0 / rho - 1`, soit 4,56 %
a `Ma = 0,3`. La planche des classifications parle de la variation de la
masse volumique d'un gaz qui s'ecoule : on la rapporte d'ordinaire a la
valeur d'arret, `(rho_0 - rho) / rho_0`, et c'est elle qui vaut 4,4 % a
`Ma = 0,3`, le chiffre que le cours et la fiche citent.
"""

from __future__ import annotations

from polytech.fluides.domain.regimes import GAMMA_DE_L_AIR
from polytech.noyau import core


def variation_relative_a_l_arret(mach: float) -> float:
    """`1 - (1 + (gamma - 1) Ma^2 / 2)^(-1/(gamma - 1))`, en ecoulement isentropique."""
    return core.variation_relative_a_l_arret(mach, GAMMA_DE_L_AIR)
