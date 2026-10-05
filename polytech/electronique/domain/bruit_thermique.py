"""Le bruit thermique de Johnson-Nyquist, d'une impedance et de deux.

Chapitre 4, partie II, planches 5 a 16. L'agitation thermique des
electrons fait apparaitre aux bornes d'une impedance une tension
gaussienne, de moyenne nulle et de densite unilaterale `4 k_B T Re[Z]` :
seule la partie reelle bruite. Le modele est blanc, donc de puissance
infinie ; la correction quantique `x / (e^x - 1)`, `x = h f / k_B T`, la
rend finie, et ne compte qu'au-dela de plusieurs centaines de gigahertz.

Un dipole bruyant se represente par son impedance sans bruit, en serie
avec sa source. Deux sources independantes se superposent en puissance :
en serie leurs densites s'ajoutent ; en parallele chacune traverse le
diviseur que forme l'autre.
"""

from __future__ import annotations

import math

from polytech.thermique.domain.rayonnement.corps_noir import BOLTZMANN, PLANCK

Bruyante = tuple[complex, float]
"""Une impedance, en ohms, et sa temperature, en kelvins."""


def densite_thermique(impedance: complex, temperature: float) -> float:
    """`4 k_B T Re[Z]`, en volts carres par hertz."""
    reelle = complex(impedance).real
    if reelle < 0:
        msg = f"partie reelle negative : {impedance} ohm -- un dipole passif dissipe"
        raise ValueError(msg)
    if temperature < 0:
        msg = f"temperature absolue negative : {temperature} K"
        raise ValueError(msg)
    return 4 * BOLTZMANN * temperature * reelle


def densite_thermique_quantique(
    resistance: float, temperature: float, frequence: float
) -> float:
    """`4 k_B T R x / (e^x - 1)`, avec `x = h f / (k_B T)`."""
    classique = densite_thermique(resistance, temperature)
    reduite = PLANCK * frequence / (BOLTZMANN * temperature)
    return classique * reduite / math.expm1(reduite) if reduite else classique


def puissance_thermique_totale(resistance: float, temperature: float) -> float:
    """`4 k_B T R (k_B T / h) pi^2 / 6` : la puissance sur toutes les frequences."""
    classique = densite_thermique(resistance, temperature)
    return classique * BOLTZMANN * temperature / PLANCK * math.pi**2 / 6


def tension_efficace_thermique(
    resistance: float, temperature: float, bande: float
) -> float:
    """`V_RMS = racine(4 k_B T R B)`."""
    if bande <= 0:
        msg = f"bande nulle ou negative : {bande} Hz"
        raise ValueError(msg)
    return math.sqrt(densite_thermique(resistance, temperature) * bande)


def densite_de_deux_impedances_en_serie(premiere: Bruyante, seconde: Bruyante) -> float:
    """`4 k_B (T_1 Re[Z_1] + T_2 Re[Z_2])`, sur l'impedance `Z_1 + Z_2`."""
    return densite_thermique(*premiere) + densite_thermique(*seconde)


def densite_de_deux_impedances_en_parallele(
    premiere: Bruyante, seconde: Bruyante
) -> float:
    """`4 k_B (T_1 Re[Z_1] |Z_2|^2 + T_2 Re[Z_2] |Z_1|^2) / |Z_1 + Z_2|^2`."""
    (z1, _), (z2, _) = premiere, seconde
    somme = abs(z1 + z2) ** 2
    if somme == 0:
        msg = "impedances opposees : leur mise en parallele n'est pas definie"
        raise ValueError(msg)
    par_la_premiere = densite_thermique(*premiere) * abs(z2) ** 2
    par_la_seconde = densite_thermique(*seconde) * abs(z1) ** 2
    return (par_la_premiere + par_la_seconde) / somme
