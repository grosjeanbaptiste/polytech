"""Le filtre de Bessel : la phase lineaire plutot que la raideur.

Chapitre 3, partie I, planches 44 a 48. `T(s) = theta_N(0) / theta_N(s)`,
ou `theta_N` est le polynome de Bessel inverse :

    theta_N(s) = somme_k (2N - k)! / (2^(N - k) k! (N - k)!) s^k.

Ainsi normalise, le retard de groupe vaut un a l'origine et y reste sur
toute la bande : c'est ce qui garde en phase les harmoniques d'un signal
carre. La perte d'insertion tend vers celle d'une gaussienne, d'ou la
bande a trois decibels `racine((2N - 1) ln 2)`.
"""

from __future__ import annotations

import math

import sympy as sp


def _coefficient(ordre: int, rang: int) -> int:
    numerateur = math.factorial(2 * ordre - rang)
    denominateur = math.factorial(rang) * math.factorial(ordre - rang)
    return numerateur // (denominateur << (ordre - rang))


def polynome_de_bessel_inverse(ordre: int, variable: sp.Expr) -> sp.Expr:
    """`theta_N(s)` ; `theta_3 = s^3 + 6 s^2 + 15 s + 15`."""
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)
    termes = (_coefficient(ordre, k) * variable**k for k in range(ordre + 1))
    return sum(termes, sp.Integer(0))


def transfert_de_bessel(variable: complex, ordre: int) -> complex:
    """`theta_N(0) / theta_N(s)`, en un point du plan complexe."""
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)
    valeur = sum(_coefficient(ordre, k) * variable**k for k in range(ordre + 1))
    return _coefficient(ordre, 0) / valeur


def poles_de_bessel(ordre: int) -> tuple[complex, ...]:
    """Les racines de `theta_N`, toutes a gauche de l'axe imaginaire."""
    variable = sp.Symbol("s")
    polynome = sp.Poly(polynome_de_bessel_inverse(ordre, variable), variable)
    return tuple(complex(racine) for racine in polynome.nroots())


def bande_a_trois_decibels_de_bessel(ordre: int) -> float:
    """`w_3dB ~ racine((2N - 1) ln 2)` ; bonne des l'ordre trois."""
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)
    return math.sqrt((2 * ordre - 1) * math.log(2))
