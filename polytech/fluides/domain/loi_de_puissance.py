"""Le profil turbulent en loi de puissance, `u / u_max = (y / R)^(1/n)`.

La fiche (article « Decider du regime ») dessine le profil turbulent
`(1 - |y - 1|)^(1/7)` : beaucoup plus plat que la parabole de Poiseuille,
puisque la turbulence brasse la quantite de mouvement sur toute la
section. L'exposant `1/7` accompagne la loi de Blasius (`Re` jusqu'a
10^5). La vitesse moyenne se deduit par integration sur la section :
`V_moy / u_max = 2 n^2 / ((n + 1) (2 n + 1))`, soit `49/60` pour `n = 7`.
"""

from __future__ import annotations

from typing import Final

EXPOSANT_DE_BLASIUS: Final = 7
"""Le `n` de la loi `1/7`, celle qui va avec `lambda = 0,316 Re^(-1/4)`."""


def _exiger_exposant(exposant: float) -> None:
    if exposant <= 0:
        msg = f"exposant non positif : n = {exposant}"
        raise ValueError(msg)


def vitesse_en_loi_de_puissance(
    distance_a_la_paroi: float,
    rayon: float,
    vitesse_maximale: float,
    exposant: float = EXPOSANT_DE_BLASIUS,
) -> float:
    """`u(y) = u_max (y / R)^(1/n)` ; leve hors de la section."""
    _exiger_exposant(exposant)
    if rayon <= 0:
        msg = f"rayon nul ou negatif : {rayon} m"
        raise ValueError(msg)
    if not 0 <= distance_a_la_paroi <= rayon:
        msg = f"hors de la section : y = {distance_a_la_paroi} pour R = {rayon}"
        raise ValueError(msg)
    return float(vitesse_maximale * (distance_a_la_paroi / rayon) ** (1 / exposant))


def rapport_vitesse_moyenne_sur_maximale(
    exposant: float = EXPOSANT_DE_BLASIUS,
) -> float:
    """`V_moy / u_max = 2 n^2 / ((n + 1) (2 n + 1))`, moyenne sur le disque."""
    _exiger_exposant(exposant)
    return 2 * exposant**2 / ((exposant + 1) * (2 * exposant + 1))
