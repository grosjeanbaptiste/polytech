"""L'equation d'onde, et la verification qu'elle offre gratuitement.

Dans le vide, les equations de Maxwell donnent
`laplacien(E) = epsilon_0 mu_0 d^2E/dt^2`, c'est-a-dire une onde de
celerite `1 / racine(epsilon_0 mu_0)`.

Cette celerite doit valoir `c`. Et comme les trois constantes sont
declarees **independamment** dans `constantes.py`, l'egalite est une
verification et non une tautologie : une faute de transcription sur l'une
des trois se verrait ici, et nulle part ailleurs.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    PERMEABILITE_DU_VIDE,
    PERMITTIVITE_DU_VIDE,
)
from polytech.electromagnetisme.domain.symboles import ABSCISSE, TEMPS


def celerite_dans_le_vide() -> float:
    """`1 / racine(epsilon_0 mu_0)`, qui doit valoir `c`."""
    return 1.0 / math.sqrt(PERMITTIVITE_DU_VIDE * PERMEABILITE_DU_VIDE)


def celerite_dans_un_milieu(permittivite: object, permeabilite: object) -> sp.Expr:
    """`v = 1 / racine(epsilon mu)` : la vitesse d'une onde dans un dielectrique.

    Le profil `E(z - v t)` ne resout l'equation d'onde que si
    `epsilon mu v^2 = 1` : c'est le resultat du changement de variable
    `w = z - v t`.
    """
    facteurs = (sp.sympify(permittivite), sp.sympify(permeabilite))
    if any(facteur.is_positive is not True for facteur in facteurs):
        msg = f"permittivite ou permeabilite nulle ou negative : {facteurs}"
        raise ValueError(msg)
    return 1 / sp.sqrt(facteurs[0] * facteurs[1])


def equation_d_onde(amplitude: sp.Expr, celerite: sp.Expr) -> sp.Expr:
    """`d^2u/dx^2 - (1/c^2) d^2u/dt^2`, qui doit valoir zero."""
    return sp.simplify(
        sp.diff(amplitude, ABSCISSE, 2) - sp.diff(amplitude, TEMPS, 2) / celerite**2,
    )


def est_une_onde(amplitude: sp.Expr, celerite: sp.Expr) -> bool:
    """Vrai si l'amplitude satisfait l'equation d'onde a cette celerite."""
    return bool(equation_d_onde(amplitude, celerite) == 0)


def onde_progressive(celerite: sp.Expr, pulsation: sp.Expr) -> sp.Expr:
    """`cos(omega (t - x/c))` : la solution de d'Alembert, forme progressive."""
    return sp.cos(pulsation * (TEMPS - ABSCISSE / celerite))


def longueur_d_onde(celerite: float, frequence: float) -> float:
    """`lambda = c / f` ; leve a frequence nulle."""
    if frequence <= 0:
        msg = f"frequence nulle ou negative : {frequence} Hz"
        raise ValueError(msg)
    return celerite / frequence


def impedance_du_vide() -> float:
    """`racine(mu_0 / epsilon_0)`, environ 377 ohms.

    Troisieme facon de recouper les deux constantes du vide : leur
    rapport donne une impedance, leur produit une celerite.
    """
    return math.sqrt(PERMEABILITE_DU_VIDE / PERMITTIVITE_DU_VIDE)
