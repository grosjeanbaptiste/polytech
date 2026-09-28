"""La pente asymptotique que la carte des poles et zeros impose.

Chaque pole retire 20 dB par decade a haute frequence, chaque zero en
ajoute autant. La carte donne donc la pente sans qu'aucun calcul de module
soit necessaire -- mais elle ne donne pas le gain, qui depend d'un facteur
que la carte ne montre pas.
"""

from __future__ import annotations

PENTE_PAR_ORDRE = 20.0
"""Decibels par decade qu'un pole ou un zero unique apporte."""


def pente_asymptotique(zeros: int, poles: int) -> float:
    """Rend la pente en dB par decade, negative quand les poles dominent."""
    if zeros < 0 or poles < 0:
        msg = f"comptes negatifs de zeros ou de poles : {zeros}, {poles}"
        raise ValueError(msg)
    return (zeros - poles) * PENTE_PAR_ORDRE
