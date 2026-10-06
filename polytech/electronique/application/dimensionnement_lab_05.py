"""Laboratoire 5 d'electronique, questions 1 a 5 : dimensionner l'ampli audio.

Une source de 25 mV, un haut-parleur de 16 ohm limite a 500 mW, la bande
audio de 20 Hz a 20 kHz, et des TLV274 de 3 MHz de produit gain-bande.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.electronique.domain.acquisition import gain_de_chaine
from polytech.electronique.domain.amplification import (
    gain_maximal_par_etage as gain_par_etage,
)
from polytech.electronique.domain.amplification import (
    gain_non_inverseur,
    nombre_d_etages,
)

SOURCE: Final = 25e-3
HAUT_PARLEUR: Final = 16.0
PUISSANCE: Final = 0.5
BANDE: Final = 20e3
PRODUIT_GAIN_BANDE: Final = 3e6
"""Celui du TLV274, d'apres le corrige officiel."""
R_MASSE: Final = 1e3
R_RETOUR: Final = 39e3
R_ENTREE: Final = 22e3
POTENTIOMETRE: Final = 100e3


def tension_efficace_maximale() -> float:
    """`P = V_eff^2 / R` : ce que le haut-parleur supporte, en volts efficaces."""
    return math.sqrt(PUISSANCE * HAUT_PARLEUR)


def crete_maximale() -> float:
    """La meme en valeur de crete, pour une sinusoide."""
    return math.sqrt(2) * tension_efficace_maximale()


def gain_maximal() -> float:
    """Ce qui porte les 25 mV de la source a cette crete."""
    return gain_de_chaine(crete_maximale(), SOURCE)


def gain_maximal_par_etage() -> float:
    """Le gain qu'un TLV274 tient sur toute la bande audio."""
    return gain_par_etage(PRODUIT_GAIN_BANDE, BANDE)


def etages() -> int:
    """Le nombre d'etages qu'il faut pour le gain maximal."""
    return nombre_d_etages(gain_maximal(), PRODUIT_GAIN_BANDE, BANDE)


def gains_realises() -> tuple[float, float]:
    """Le non-inverseur d'entree, puis l'inverseur a son maximum (en module)."""
    return gain_non_inverseur(R_MASSE, R_RETOUR), POTENTIOMETRE / R_ENTREE


def bandes_des_etages() -> tuple[float, float]:
    """La bande de chaque etage : `GBW` divise par son gain de bruit."""
    premier, second = gains_realises()
    return PRODUIT_GAIN_BANDE / premier, PRODUIT_GAIN_BANDE / (1 + second)


def courant_maximal() -> float:
    """Ce que la charge tire a la crete maximale."""
    return crete_maximale() / HAUT_PARLEUR
