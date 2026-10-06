"""TP 7 d'electronique, exercice 4.1 : un Sallen-Key a composants imprecis.

La cellule est dimensionnee en valeurs normalisees par la methode a
sensibilite minimale (`methodes_sallen_key`), puis denormalisee. Le
module ajoute ce que l'enonce demande d'ignorer : avec `Q = 1`, la
pulsation propre n'est pas la coupure a 3 dB, et la reponse a une bosse
-- que la mise en cascade de trois cellules triple.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from polytech.electronique.domain import reponse_du_biquad as rb
from polytech.electronique.domain.methodes_sallen_key import (
    Methode,
    cellule_normalisee,
)
from polytech.electronique.domain.sections_actives import biquads_de_butterworth
from polytech.electronique.domain.sensibilite_sallen_key import (
    sensibilites_de_la_qualite,
)

if TYPE_CHECKING:
    from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas

METHODE_DU_4_1: Final = Methode.SENSIBILITE_MINIMALE
FREQUENCE_DU_4_1: Final = 200e3
CAPACITE_MAXIMALE_DU_4_1: Final = 10e-6
QUALITE_DU_4_1: Final = 1.0
_PASSIFS: Final = ("r1", "r2", "c1", "c2")


def facteur_pour_les_resistances(resistance: float = 100.0) -> float:
    """4.1.2 : la methode 4 part de 1 ohm ; `gamma_0` est la resistance visee."""
    return resistance / 1.0


def facteur_pour_les_capacites(maximale: float = CAPACITE_MAXIMALE_DU_4_1) -> float:
    """4.1.3 : `gamma_0 >= 2 Q / (w_c C_max)`, la plus grande capacite etant `C_1`."""
    pulsation = 2 * math.pi * FREQUENCE_DU_4_1
    return 2 * QUALITE_DU_4_1 / (pulsation * maximale)


def cellule_du_4_1(resistance: float = 100.0) -> SallenKeyPasseBas:
    """4.1.4 et 4.1.6 : la cellule a suiveur, pour des resistances donnees."""
    normalisee = cellule_normalisee(METHODE_DU_4_1, QUALITE_DU_4_1, 1)
    return normalisee.denormalisee(
        coupure=2 * math.pi * FREQUENCE_DU_4_1,
        impedance=facteur_pour_les_resistances(resistance),
    )


def coupure_reelle_du_4_1(cellules: int = 1) -> float:
    """La frequence a -3 dB de `cellules` cellules en cascade, en hertz."""
    rapport = rb.coupure_a_trois_decibels(QUALITE_DU_4_1, cellules=cellules)
    return FREQUENCE_DU_4_1 * rapport


def bosse_du_4_1(cellules: int) -> float:
    """Le depassement du gain continu, en decibels, de la cascade."""
    return cellules * 20 * math.log10(rb.surtension(QUALITE_DU_4_1))


def cellules_de_butterworth(
    ordre: int = 6, resistance: float = 100.0
) -> tuple[SallenKeyPasseBas, ...]:
    """4.1.5 autrement : les cellules d'un vrai Butterworth, `Q` croissants.

    Trois cellules *identiques* de `Q = 1` empilent trois bosses. Trois
    cellules de `Q = 1 / b_1` differents, les `b_1` etant ceux du
    prototype, donnent la meme pente sans aucune bosse.
    """
    amortissements = sorted(biquads_de_butterworth(ordre), reverse=True)
    return tuple(
        cellule_normalisee(METHODE_DU_4_1, 1 / b1, 1).denormalisee(
            coupure=2 * math.pi * FREQUENCE_DU_4_1,
            impedance=facteur_pour_les_resistances(resistance),
        )
        for b1 in amortissements
    )


def pires_sensibilites_passives(qualite: float) -> dict[Methode, float]:
    """La plus grande `|S|` de `Q` aux resistances et capacites, par methode."""
    pires = {}
    for methode in Methode:
        valeurs = sensibilites_de_la_qualite(cellule_normalisee(methode, qualite, 1))
        pires[methode] = max(abs(float(valeurs[nom])) for nom in _PASSIFS)
    return pires
