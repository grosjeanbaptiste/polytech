"""TP 7 d'electronique, exercices 3.1 a 3.4 : quatre filtres LC en echelle.

La marche est celle du cours et du corrige officiel : le prototype
normalise (table des `g_i`), la mise a l'echelle des impedances, puis la
transformation de frequence. Chaque filtre est rendu avec ses
terminaisons : `EchelleTerminee` le remonte en circuit et dit son module.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.electronique.domain.chebyshev import (
    charge_de_chebyshev,
    elements_de_chebyshev,
)
from polytech.electronique.domain.conversions_d_elements import (
    Branche,
    changer_la_coupure,
    en_passe_bande,
    en_passe_haut,
)
from polytech.electronique.domain.echelle_de_branches import EchelleTerminee
from polytech.electronique.domain.echelle_lc import elements_de_butterworth
from polytech.electronique.domain.passe_bande import bords_du_passe_bande
from polytech.electronique.domain.reponse_de_chebyshev import coupure_a_trois_decibels

ONDULATION: Final = 0.1
"""`L_Ar`, en decibels, des trois filtres de Chebyshev de la seance."""
_DEUX_PI: Final = 2 * math.pi
CENTRE: Final = _DEUX_PI * 900e6
LARGEUR: Final = _DEUX_PI * 200e6
COUPURE_DU_PASSE_HAUT: Final = _DEUX_PI * 110e6


def prototype(
    elements: tuple[float, ...], resistance: float, *, serie_d_abord: bool
) -> tuple[Branche, ...]:
    """L'echelle normalisee, mise a l'echelle : `L -> R L`, `C -> C / R`."""
    branches = []
    en_serie = serie_d_abord
    for valeur in elements:
        if en_serie:
            branches.append(Branche(inductance=resistance * valeur))
        else:
            branches.append(Branche(capacite=valeur / resistance))
        en_serie = not en_serie
    return tuple(branches)


def butterworth_dix_megahertz() -> EchelleTerminee:
    """Exercice 3.1 : ordre 2, 10 MHz, 50 ohm, inductance d'abord."""
    echelle = prototype(elements_de_butterworth(2), 50.0, serie_d_abord=True)
    branches = tuple(changer_la_coupure(b, _DEUX_PI * 10e6) for b in echelle)
    return EchelleTerminee(branches, serie_d_abord=True, source=50.0, charge=50.0)


def charges_de_chebyshev(ordre_maximal: int) -> dict[int, float]:
    """Exercice 3.1.2 : la charge normalisee `g_(N+1)` de chaque ordre."""
    ordres = range(1, ordre_maximal + 1)
    return {n: charge_de_chebyshev(ONDULATION, n) for n in ordres}


def passe_bas_d_ordre_trois(*, chebyshev: bool) -> EchelleTerminee:
    """Exercice 3.2 : `10^5 rad/s`, 32 ohm, capacite d'abord -- une inductance."""
    g = (
        elements_de_chebyshev(ONDULATION, 3)
        if chebyshev
        else elements_de_butterworth(3)
    )
    echelle = prototype(g, 32.0, serie_d_abord=False)
    branches = tuple(changer_la_coupure(b, 1e5) for b in echelle)
    return EchelleTerminee(branches, serie_d_abord=False, source=32.0, charge=32.0)


def rapport_de_coupure_du_chebyshev(ordre: int) -> float:
    """`w_3dB / w_c` d'un Chebyshev de 0,1 dB : sa coupure borne l'ondulation."""
    epsilon = math.sqrt(10 ** (ONDULATION / 10) - 1)
    return coupure_a_trois_decibels(ordre, epsilon)


def passe_bande(centre: float = CENTRE) -> EchelleTerminee:
    """Exercice 3.3 : Chebyshev d'ordre 3, 50 ohm, branche serie d'abord."""
    echelle = prototype(elements_de_chebyshev(ONDULATION, 3), 50.0, serie_d_abord=True)
    branches = tuple(en_passe_bande(b, centre, LARGEUR) for b in echelle)
    return EchelleTerminee(branches, serie_d_abord=True, source=50.0, charge=50.0)


def bords_de_la_bande(centre: float = CENTRE) -> tuple[float, float]:
    """Les deux pulsations ou le passe-bande quitte son ondulation."""
    return bords_du_passe_bande(centre, LARGEUR)


def centre_geometrique() -> float:
    """`racine(w_1 w_2)` de la bande [800, 1000] MHz que l'enonce demande."""
    return _DEUX_PI * math.sqrt(800e6 * 1000e6)


def passe_haut(*, prototype_en_serie: bool = True) -> EchelleTerminee:
    """Exercice 3.4 : Chebyshev d'ordre 2, 110 MHz, source de 50 ohm.

    `g_3` est une resistance apres une capacite en derivation, une
    conductance apres une inductance serie : la charge depend donc de la
    topologie retenue pour le prototype.
    """
    g = elements_de_chebyshev(ONDULATION, 2)
    echelle = prototype(g, 50.0, serie_d_abord=prototype_en_serie)
    derniere = charge_de_chebyshev(ONDULATION, 2)
    charge = 50.0 * derniere if prototype_en_serie else 50.0 / derniere
    branches = tuple(en_passe_haut(b, COUPURE_DU_PASSE_HAUT) for b in echelle)
    return EchelleTerminee(
        branches, serie_d_abord=prototype_en_serie, source=50.0, charge=charge
    )


def perte_a_huit_cents_megahertz() -> float:
    """Ce que le filtre centre sur 900 MHz perd, en decibels, au bord demande."""
    return -20 * math.log10(2 * passe_bande().module(_DEUX_PI * 800e6))
