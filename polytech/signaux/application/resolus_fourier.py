"""Exercices resolus, 0.12 a 0.14 : spectres de raies et modulation.

Repeter un signal toutes les `T0` secondes ne garde de son spectre que
des raies, aux multiples de `1/T0`. Le multiplier par un cosinus deplace
son spectre de part et d'autre de la porteuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.signaux.application import recueil_spectres
from polytech.signaux.domain.frequence import FREQUENCE as F
from polytech.signaux.domain.frequence import INSTANT, rect, transformee_en_frequence
from polytech.signaux.domain.raies import raie

if TYPE_CHECKING:
    from collections.abc import Sequence

DEMI: Final = sp.Rational(1, 2)
LARGEUR_DE_LA_PORTE: Final = sp.Rational(1, 5)
"""`rect(5t)` vaut un pendant un cinquieme de seconde."""
PERIODE_DE_LA_PORTE: Final = sp.Integer(1)
F0: Final = sp.Symbol("f_0", positive=True)
PORTEUSE_D_ESSAI: Final = 3
"""Une frequence porteuse, en hertz, pour chiffrer la verification de 0.14."""


def raies_de_la_porte(rangs: Sequence[int]) -> list[sp.Expr]:
    """0.12 : les raies de ces rangs, pour `rect(5t)` repetee chaque seconde."""
    image = transformee_en_frequence(rect(INSTANT / LARGEUR_DE_LA_PORTE))
    return [raie(image, PERIODE_DE_LA_PORTE, rang) for rang in rangs]


def raie_de_la_porte(rang: int) -> sp.Expr:
    """La raie d'un seul rang."""
    return raies_de_la_porte([rang])[0]


def niveau_de_la_raie_centrale() -> float:
    """0.13 : la raie de rang zero de la rampe repetee, en decibels."""
    return recueil_spectres.niveau_de_la_raie(0)


def ecart_des_raies() -> sp.Expr:
    """0.13 : `1/T0`, en hertz."""
    return recueil_spectres.ecart_des_raies()


def _spectre_module(porteuse: sp.Expr) -> sp.Expr:
    """0.14 : la moitie du spectre de `sinc(t)`, deplacee en `+f0` et en `-f0`."""
    return (rect(F - porteuse) + rect(F + porteuse)) / 2


def spectre_module(frequence: object, porteuse: object = PORTEUSE_D_ESSAI) -> sp.Expr:
    """La valeur du spectre du signal module, en cette frequence."""
    return _spectre_module(sp.sympify(porteuse)).subs(F, frequence)


def les_lobes_se_recouvrent(porteuse: object) -> bool:
    """Vrai si la porteuse est trop basse : les portes, larges de un, se touchent."""
    return bool(2 * sp.sympify(porteuse) <= 1)


def bords_du_spectre_module() -> tuple[sp.Expr, sp.Expr]:
    """Les bords de la porte de droite : la largeur du spectre de `sinc(t)` est un."""
    return F0 - DEMI, F0 + DEMI


def signal_depuis_le_spectre_module(instant: float) -> float:
    """L'inverse de Fourier du spectre, pour la porteuse d'essai, en un instant."""
    noyau = sp.exp(2 * sp.pi * sp.I * F * instant)
    centre = sp.Integer(PORTEUSE_D_ESSAI)
    lobes = ((centre - DEMI, centre + DEMI), (-centre - DEMI, -centre + DEMI))
    somme = sum(sp.integrate(noyau / 2, (F, bas, haut)) for bas, haut in lobes)
    return float(sp.re(sp.N(somme)))
