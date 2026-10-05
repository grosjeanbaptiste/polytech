"""La resistance d'un conducteur en courant sinusoidal.

Le syllabus d'electricite generale (16.22.1). Quand l'effet pelliculaire
est intense, tout se passe comme si le courant circulait dans une coque
peripherique dont l'epaisseur est la profondeur de penetration `delta` :
la resistance effective d'un troncon vaut `rho l / (perimetre . delta)`
(16.22.2), le perimetre etant « ou mieux » celui du milieu de la coque.

Pour un fil de section circulaire, le rapport de la resistance en
sinusoidal a la resistance en continu ne depend que de `a / delta` : le
syllabus en donne quatre valeurs, de 1,001 a 100. Elles sortent du profil
de courant `J_0(k r)`, solution de l'equation de diffusion (15.14.3) pour
`k^2 = -j omega mu / rho`, soit `k = (1 - j) / delta` : le rapport est la
partie reelle de `(k a / 2) J_0(k a) / J_1(k a)`.
"""

from __future__ import annotations

import mpmath


def resistance_effective(
    resistivite: float, longueur: float, perimetre: float, peau: float
) -> float:
    """`R_CA = rho l / (perimetre . delta)` (16.22.2), en ohms."""
    if perimetre <= 0 or peau <= 0:
        msg = f"coque sans section : perimetre {perimetre} m, epaisseur {peau} m"
        raise ValueError(msg)
    return resistivite * longueur / (perimetre * peau)


def rapport_des_resistances_d_un_fil(rayon_sur_peau: float) -> float:
    """`R_CA / R_CC` d'un fil circulaire, en fonction de `a / delta`."""
    if rayon_sur_peau <= 0:
        msg = f"rayon nul ou negatif : a / delta = {rayon_sur_peau}"
        raise ValueError(msg)
    argument = mpmath.mpc(1, -1) * rayon_sur_peau
    bessel = mpmath.besselj(0, argument) / mpmath.besselj(1, argument)
    return float(mpmath.re(argument / 2 * bessel))
