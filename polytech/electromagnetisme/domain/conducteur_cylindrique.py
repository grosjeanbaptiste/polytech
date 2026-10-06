"""Le conducteur cylindrique, et la densite de courant qui s'y etablit.

Dans un metal, la loi d'Ampere se simplifie : le courant de deplacement
`eps_0 dE/dt` y est au courant de conduction `sigma E` ce que `eps_0 w`
est a `sigma`. Pour le cuivre le rapport `sigma / (eps_0 w)` depasse
`5e18 / w` : il faut atteindre 800 THz pour que le second terme pese un
millieme du premier. D'ou `rot B = mu_0 sigma E`.

Un courant **constant** `I` dans un fil de rayon `a` se repartit alors
uniformement : `E_0 = I / (sigma pi a^2)` le long de l'axe, et `B_phi =
mu_0 sigma E_0 r / 2` autour. Un courant **variable** ne le peut pas : ce
champ magnetique varie, et la loi de Faraday exige alors un champ
electrique qui depende du rayon. C'est l'effet de peau.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    PERMEABILITE_DU_VIDE,
    PERMITTIVITE_DU_VIDE,
)


def rapport_conduction_deplacement(
    conductivite: float, pulsation: float, *, permittivite: float = PERMITTIVITE_DU_VIDE
) -> float:
    """`sigma / (eps w)` : le courant de conduction rapporte a celui de deplacement."""
    if pulsation <= 0:
        msg = f"pulsation nulle ou negative : {pulsation} rad/s"
        raise ValueError(msg)
    if permittivite <= 0:
        msg = f"permittivite nulle ou negative : {permittivite} F/m"
        raise ValueError(msg)
    return conductivite / (permittivite * pulsation)


def champ_electrique_uniforme(
    courant: sp.Expr, conductivite: sp.Expr, rayon: sp.Expr
) -> sp.Expr:
    """`E_0 = I / (sigma pi a^2)` : le champ d'un courant constant dans le fil."""
    if conductivite == 0 or rayon == 0:
        msg = "conductivite ou rayon nul : le fil ne porte aucun courant fini"
        raise ValueError(msg)
    return courant / (conductivite * sp.pi * rayon**2)


def champ_magnetique_interne(
    distance: sp.Expr, champ: sp.Expr, conductivite: sp.Expr
) -> sp.Expr:
    """`B_phi = mu_0 sigma E_0 r / 2`, a la distance `r` de l'axe, dans le fil."""
    return sp.Float(PERMEABILITE_DU_VIDE) * conductivite * champ * distance / 2
