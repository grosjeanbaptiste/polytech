"""La convection naturelle, par Grashof, Rayleigh, et la lame de fluide.

Le formulaire donne la lame de fluide entre deux plaques en clair, avec
ses trois regimes :

* `Gr_b Pr < 1400` : `Nu_b = 1` -- le fluide ne bouge pas, il conduit ;
* `6000 < Gr_b Pr < 10^6` : `Nu_b = 0,11 (Gr_b Pr)^0,29` ;
* `10^6 < Gr_b Pr < 10^8` : `Nu_b = 0,40 (Gr_b Pr)^0,20`.

Le **trou** entre 1400 et 6000 est dans le formulaire, et il n'est pas
comble ici : c'est une zone de transition que le cours ne correle pas, et
y interpoler serait inventer une correlation. La garde le dit.
"""

from __future__ import annotations

from typing import Final

PESANTEUR: Final = 9.81
"""Acceleration de la pesanteur, en m/s^2."""

CONDUCTION_PURE: Final = 1400.0
"""En dessous, le formulaire donne `Nu = 1` : la lame conduit sans bouger."""

DEBUT_DU_PREMIER_REGIME: Final = 6000.0
FIN_DU_PREMIER_REGIME: Final = 1e6
FIN_DU_SECOND_REGIME: Final = 1e8


def grashof(
    dilatation: float,
    ecart: float,
    longueur: float,
    viscosite_cinematique: float,
) -> float:
    """`Gr = g beta Delta T L^3 / nu^2`."""
    if viscosite_cinematique <= 0:
        msg = f"viscosite cinematique nulle ou negative : {viscosite_cinematique}"
        raise ValueError(msg)
    numerateur = PESANTEUR * dilatation * abs(ecart) * longueur**3
    return numerateur / viscosite_cinematique**2


def rayleigh(grashof_local: float, prandtl_local: float) -> float:
    """`Ra = Gr Pr`, le produit dont depend le regime."""
    return grashof_local * prandtl_local


def nusselt_lame_de_fluide(rayleigh_local: float) -> float:
    """La correlation du formulaire ; leve dans la zone qu'il ne correle pas."""
    if rayleigh_local < CONDUCTION_PURE:
        return 1.0
    if rayleigh_local < DEBUT_DU_PREMIER_REGIME:
        borne = f"{CONDUCTION_PURE} < {rayleigh_local:.3g} < {DEBUT_DU_PREMIER_REGIME}"
        msg = f"zone de transition non correlee par le formulaire : {borne}"
        raise ValueError(msg)
    if rayleigh_local <= FIN_DU_PREMIER_REGIME:
        return float(0.11 * rayleigh_local**0.29)
    if rayleigh_local <= FIN_DU_SECOND_REGIME:
        return float(0.40 * rayleigh_local**0.20)
    msg = (
        f"Gr Pr = {rayleigh_local:.3g} depasse la derniere plage du "
        f"formulaire ({FIN_DU_SECOND_REGIME:.0e})"
    )
    raise ValueError(msg)
