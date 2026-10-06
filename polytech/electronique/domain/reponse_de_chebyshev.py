"""Le filtre de Chebyshev de type I : module, poles, transfert.

Chapitre 3, partie I, planches 33 a 36. `|T(j w)| = 1 / racine(1 + eps^2
T_N(w)^2)`, pour une coupure normalisee a 1 rad/s : le polynome de
Chebyshev `T_N` oscille entre -1 et 1 dans la bande, d'ou l'ondulation,
et croit comme un cosinus hyperbolique au-dela, d'ou la raideur.

Les poles sont `-p_i`, avec `p_i = j cos(asin(j eta) + (2i - 1) pi / 2N)`
et `eta = sinh(asinh(1 / eps) / N)` : ils sont sur une demi-ellipse, la
ou ceux de Butterworth sont sur un demi-cercle.
"""

from __future__ import annotations

import cmath
import math


def polynome_de_chebyshev(ordre: int, pulsation: float) -> float:
    """`cos(N acos w)` si `|w| <= 1`, `cosh(N acosh |w|)` au-dela, au signe pres."""
    if ordre < 0:
        msg = f"ordre negatif : {ordre}"
        raise ValueError(msg)
    if abs(pulsation) <= 1:
        return math.cos(ordre * math.acos(pulsation))
    signe = -1 if pulsation < 0 and ordre % 2 else 1
    return signe * math.cosh(ordre * math.acosh(abs(pulsation)))


def module_de_chebyshev(pulsation: float, ordre: int, epsilon: float) -> float:
    """`1 / racine(1 + eps^2 T_N(w)^2)`."""
    polynome = polynome_de_chebyshev(ordre, pulsation)
    return 1 / math.sqrt(1 + (epsilon * polynome) ** 2)


def _eta(ordre: int, epsilon: float) -> float:
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = f"parametre d'ondulation nul ou negatif : {epsilon}"
        raise ValueError(msg)
    return math.sinh(math.asinh(1 / epsilon) / ordre)


def poles_de_chebyshev(ordre: int, epsilon: float) -> tuple[complex, ...]:
    """Les `-p_i` de la planche 34, a gauche de l'axe imaginaire."""
    base = cmath.asin(1j * _eta(ordre, epsilon))
    return tuple(
        -1j * cmath.cos(base + (2 * rang - 1) * math.pi / (2 * ordre))
        for rang in range(1, ordre + 1)
    )


def transfert_de_chebyshev(variable: complex, ordre: int, epsilon: float) -> complex:
    """`prod racine(eta^2 + sin^2(i pi / N)) / prod (s + p_i)`."""
    eta = _eta(ordre, epsilon)
    numerateur = math.prod(
        math.sqrt(eta**2 + math.sin(rang * math.pi / ordre) ** 2)
        for rang in range(1, ordre + 1)
    )
    denominateur = complex(1)
    for pole in poles_de_chebyshev(ordre, epsilon):
        denominateur *= variable - pole
    return numerateur / denominateur


def coupure_a_trois_decibels(ordre: int, epsilon: float) -> float:
    """`cosh(acosh(1 / eps) / N)` : ou le module tombe a `1 / racine(2)`.

    La pulsation unite du prototype borne l'*ondulation*, pas les 3 dB :
    `eps T_N(w) = 1` se resout au-dela, tant que l'ondulation elle-meme
    reste sous 3 dB (`eps <= 1`).
    """
    _eta(ordre, epsilon)
    if epsilon > 1:
        msg = f"ondulation superieure a 3 dB, coupure dans la bande : {epsilon}"
        raise ValueError(msg)
    return math.cosh(math.acosh(1 / epsilon) / ordre)
