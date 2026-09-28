"""Le bruit thermique d'une resistance (bruit de Johnson-Nyquist).

La densite spectrale de puissance se donne sous deux conventions. La
version *unilaterale*, celle du cours, ne compte que les frequences
positives et vaut donc le double de la bilaterale : c'est la distinction
que le mot « one-sided » de l'enonce recouvre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp

FACTEUR_UNILATERAL = 4
"""Deux pour la convention unilaterale, deux pour le theoreme de Nyquist."""


def densite_spectrale_unilaterale(
    resistance: sp.Expr,
    temperature: sp.Expr,
    constante_de_boltzmann: sp.Expr,
) -> sp.Expr:
    """Rend `4 k_B R T`, en volts carres par hertz."""
    return FACTEUR_UNILATERAL * constante_de_boltzmann * resistance * temperature
