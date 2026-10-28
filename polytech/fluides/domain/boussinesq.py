"""L'approximation de Boussinesq : rho varie avec T, mais seulement en flottabilite.

Dans les equations de Navier-Stokes d'un fluide chauffe, la masse
volumique `rho = rho_0 (1 - beta (T - T_0))` n'est gardee variable que
dans le terme de pesanteur ; partout ailleurs (inertie, continuite) on
prend `rho_0`. La force de flottabilite par unite de volume est alors
`rho_0 g beta (T - T_0)`, vers le haut pour un fluide plus chaud.
Licite tant que `beta |T - T_0|` reste petit devant un.
"""

from __future__ import annotations

from polytech.fluides.domain.grandeurs import PESANTEUR


def _exiger_masse_volumique(masse_volumique: float) -> None:
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)


def masse_volumique_de_boussinesq(
    reference: float,
    coefficient: float,
    temperature: float,
    temperature_reference: float,
) -> float:
    """`rho = rho_0 (1 - beta (T - T_0))` ; leve si elle s'annule."""
    _exiger_masse_volumique(reference)
    rho = reference * (1 - coefficient * (temperature - temperature_reference))
    if rho <= 0:
        msg = f"Boussinesq hors domaine : rho = {rho} kg/m^3 (beta dT >= 1)"
        raise ValueError(msg)
    return float(rho)


def flottabilite_volumique(
    reference: float,
    coefficient: float,
    temperature: float,
    temperature_reference: float,
) -> float:
    """`(rho_0 - rho) g = rho_0 g beta (T - T_0)`, en N/m^3, vers le haut."""
    _exiger_masse_volumique(reference)
    return float(
        reference * PESANTEUR * coefficient * (temperature - temperature_reference)
    )


def boussinesq_est_licite(
    coefficient: float, ecart_de_temperature: float, *, tolerance: float
) -> bool:
    """`beta |dT| <= tolerance` : la variation de rho reste negligeable ailleurs."""
    if coefficient <= 0 or tolerance <= 0:
        msg = f"coefficient ou tolerance non positif : {coefficient}, {tolerance}"
        raise ValueError(msg)
    return coefficient * abs(ecart_de_temperature) <= tolerance
