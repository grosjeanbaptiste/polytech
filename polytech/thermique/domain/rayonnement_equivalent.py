"""Le rayonnement ecrit comme un coefficient de transmission.

Syllabus de thermique, tome 2, deuxieme partie, section 1.3.
`K_r = phi_r / Delta t` range le rayonnement dans un schema de
resistances, a cote de la convection, et permet de juger d'un calcul
rapide s'il compte.

Tout passe par une fonction de deux temperatures absolues,

    K_sigma(T_1, T_2) = sigma (T_1^4 - T_2^4) / (T_1 - T_2) ~ 4 sigma T_12^3,

que le syllabus met en table : 5,7 W/m^2 K a 20 degres, 104,8 a 500,
468 a 1000. Entre surfaces grises, `K_r = F K_sigma` (1.7) ; pour un gaz
emetteur, (1.9) et sa forme linearisee (1.11).
"""

from __future__ import annotations

from polytech.thermique.domain.rayonnement.corps_noir import STEFAN_BOLTZMANN


def _exiger_des_kelvins(*temperatures: float) -> None:
    for temperature in temperatures:
        if temperature <= 0:
            msg = f"temperature absolue nulle ou negative : {temperature} K"
            raise ValueError(msg)


def coefficient_sigma(premiere: float, seconde: float) -> float:
    """`K_sigma`, en W/(m^2 K) ; a temperatures egales, sa limite `4 sigma T^3`."""
    _exiger_des_kelvins(premiere, seconde)
    if premiere == seconde:
        return 4 * STEFAN_BOLTZMANN * premiere**3
    return STEFAN_BOLTZMANN * (premiere**4 - seconde**4) / (premiere - seconde)


def coefficient_entre_surfaces(facteur: float, paroi: float, autre: float) -> float:
    """(1.7) et (1.8) : `K_r = F K_sigma`, `F` valant `epsilon` face a un local."""
    if not 0 <= facteur <= 1:
        msg = f"facteur de rayonnement hors de [0, 1] : {facteur}"
        raise ValueError(msg)
    return facteur * coefficient_sigma(paroi, autre)


def coefficient_du_gaz(
    emissivite_de_paroi: float,
    *,
    emission: float,
    absorption: float,
    gaz: float,
    paroi: float,
) -> float:
    """(1.9) : `(eps_p + 1)/2 sigma (eps_g T_g^4 - alpha_gp T_p^4) / (T_g - T_p)`."""
    _exiger_des_kelvins(gaz, paroi)
    if gaz == paroi:
        msg = f"le gaz et la paroi sont a la meme temperature : {gaz} K"
        raise ValueError(msg)
    echange = emission * gaz**4 - absorption * paroi**4
    return (emissivite_de_paroi + 1) / 2 * STEFAN_BOLTZMANN * echange / (gaz - paroi)


def coefficient_du_gaz_linearise(
    emissivite_de_paroi: float, emission: float, moyenne: float
) -> float:
    """(1.11) : `2 (eps_p + 1) eps_g sigma T_gp^3`, ou `alpha_gp` vaut `eps_g`."""
    return (
        (emissivite_de_paroi + 1) / 2 * emission * coefficient_sigma(moyenne, moyenne)
    )
