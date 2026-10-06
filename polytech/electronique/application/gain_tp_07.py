"""TP 7 d'electronique, exercice 4.2 : un Sallen-Key qui amplifie.

Le gain continu demande, 5/2, ne laisse qu'une methode -- celle des
elements egaux, ou `mu = 3 - 1/Q` -- et fixe donc `Q = 2`. Ce `Q` a un
prix que l'enonce demande d'ignorer et que ce module chiffre : la
reponse monte a plus de deux fois le gain continu pres de `f_0`, et la
coupure a 3 dB est une fois et demie plus haut que `f_0`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from polytech.electronique.domain import reponse_du_biquad as rb
from polytech.electronique.domain.methodes_sallen_key import (
    Methode,
    cellule_normalisee,
    methodes_a_gain_fixe,
    qualite_des_elements_egaux,
    resistances_de_gain,
)

if TYPE_CHECKING:
    from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas

FREQUENCE: Final = 10e3
CAPACITE_MAXIMALE: Final = 0.1e-6
RESISTANCE_MINIMALE: Final = 1e3
ALIMENTATION: Final = 5.0
CRETE_D_ENTREE: Final = 2.0
_PULSATION: Final = 2 * math.pi * FREQUENCE


def gain() -> float:
    """4.2.1 : 5 V de crete en sortie pour 2 V en entree."""
    return ALIMENTATION / CRETE_D_ENTREE


def methode(gain_vise: float | None = None) -> tuple[Methode, float]:
    """4.2.2 et 4.2.3 : la seule methode qui tienne ce gain, et son `Q`.

    Les methodes 2 a 4 n'ont chacune qu'un gain. Si aucune n'a celui-ci,
    il ne reste que la premiere, qui l'atteint en imposant `Q`.
    """
    vise = gain() if gain_vise is None else gain_vise
    autres = methodes_a_gain_fixe(vise)
    if autres:
        noms = ", ".join(m.name for m in autres)
        msg = f"la methode 1 n'est pas seule a tenir le gain {vise} : {noms}"
        raise ValueError(msg)
    return Methode.ELEMENTS_EGAUX, qualite_des_elements_egaux(vise)


def resistance_normalisee() -> float:
    """4.2.4 : `R = 1 / w_c`, en ohms, avant toute mise a l'echelle."""
    return 1 / _PULSATION


def facteurs_d_echelle() -> tuple[float, float]:
    """Les deux bornes de `gamma_0` : resistances d'au moins 1 kohm, capacites."""
    return RESISTANCE_MINIMALE / resistance_normalisee(), 1 / CAPACITE_MAXIMALE


def cellule() -> SallenKeyPasseBas:
    """La cellule finale, au plus petit `gamma_0` qui tienne les deux bornes.

    La methode est appliquee a `w_c` meme, comme dans le corrige : il ne
    reste qu'une mise a l'echelle des impedances.
    """
    seule, qualite = methode()
    a_la_pulsation = cellule_normalisee(seule, qualite, _PULSATION)
    return a_la_pulsation.denormalisee(coupure=1, impedance=max(facteurs_d_echelle()))


def pont_officiel() -> tuple[float, float]:
    """`(R_a, R_b)` du corrige : le plus petit pont permis, 1 kohm et 1,5 kohm."""
    return RESISTANCE_MINIMALE, (gain() - 1) * RESISTANCE_MINIMALE


def pont_compense() -> tuple[float, float]:
    """`(R_a, R_b)` tel que `R_a // R_b = R_1 + R_2` (planche 30 du cours)."""
    finale = cellule()
    return resistances_de_gain(gain(), float(finale.r1 + finale.r2))


def bande_plate(ecart_en_decibels: float = 1.0) -> float:
    """La frequence jusqu'a laquelle le gain reste a cet ecart du continu."""
    _, qualite = methode()
    return FREQUENCE * rb.premiere_pulsation_d_ecart(qualite, ecart_en_decibels)


def sommet() -> tuple[float, float]:
    """La frequence du sommet de la reponse, et le gain qu'elle y atteint."""
    _, qualite = methode()
    return FREQUENCE * rb.pulsation_de_resonance(qualite), gain() * rb.surtension(
        qualite
    )


def crete_au_sommet() -> float:
    """La crete de sortie qu'un signal de 2 V place au sommet demanderait."""
    return CRETE_D_ENTREE * sommet()[1]


def coupure_reelle() -> float:
    """La frequence a -3 dB de la cellule, `f_0` valant 10 kHz."""
    _, qualite = methode()
    return FREQUENCE * rb.coupure_a_trois_decibels(qualite)


def frequence_propre_pour_couper_a_dix_kilohertz() -> float:
    """La `f_0` qu'il faudrait pour que les 3 dB tombent sur 10 kHz."""
    _, qualite = methode()
    return FREQUENCE / rb.coupure_a_trois_decibels(qualite)
