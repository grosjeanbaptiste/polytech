"""Des sources de bruit egales dans un auditoire : decibels et amplitudes.

Les puissances de bruit s'ajoutent (`P = N P_1`) : le rapport signal sur
bruit varie de `10 log10(N_ref / N)` quand le nombre de sources change.
"""

from __future__ import annotations

import math


def rsb_avec(
    sources: int, *, de_reference: tuple[int, float], amplitude: float = 1.0
) -> float:
    """Le RSB (dB) avec `sources` etudiants, connu pour `(N_ref, RSB_ref)`."""
    reference, rsb = de_reference
    if sources < 1 or reference < 1 or amplitude <= 0:
        msg = f"donnees invalides : {sources}, {reference}, {amplitude}"
        raise ValueError(msg)
    return rsb + 10 * math.log10(reference / sources) + 20 * math.log10(amplitude)


def etudiants_a_garder(sources: int, rsb: float, rsb_vise: float) -> int:
    """Le nombre de sources qui donne `rsb_vise` ; leve s'il n'est pas entier."""
    exact = sources / 10 ** ((rsb_vise - rsb) / 10)
    if abs(exact - round(exact)) > 1e-2 * exact:
        msg = f"{exact:.3f} sources : aucun nombre entier ne donne {rsb_vise} dB"
        raise ValueError(msg)
    return round(exact)


def facteur_d_amplitude(ecart_en_decibels: float) -> float:
    """`10^(ecart/20)` : le facteur sur l'amplitude pour gagner cet ecart."""
    return 10 ** (ecart_en_decibels / 20)


def facteur_pour_atteindre(rsb_actuel: float, rsb_vise: float) -> float:
    """Le facteur d'amplitude, quel que soit le nombre de sources restantes."""
    return facteur_d_amplitude(rsb_vise - rsb_actuel)
