"""Une cheminee a parois en couches, traversee par des fumees qui se refroidissent.

Le fluide d'un cote, l'air de l'autre : la cheminee est un echangeur dont la
paroi est a temperature inconnue, d'ou le point fixe sur la paroi exterieure
(le coefficient de rayonnement en depend). La sortie vient du rendement d'un
echangeur dont un cote est a temperature constante (`R = 0`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.thermique.domain.conduction.parois import Tube
from polytech.thermique.domain.echangeurs_a_courants_paralleles import Sens
from polytech.thermique.domain.rendement_d_echangeur import rendement

if TYPE_CHECKING:
    from collections.abc import Callable

ITERATIONS = 100
TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Cheminee:
    """Geometrie, couches `(epaisseur, conductivite)` de l'interieur a l'exterieur."""

    hauteur: float
    diametre_interieur: float
    couches: tuple[tuple[float, float], ...]
    capacite_des_fumees: float
    entree: float
    ambiance: float
    coefficient_interieur: float


@dataclass(frozen=True, slots=True)
class Etat:
    """Sortie des fumees, flux et temperatures moyennes des deux faces."""

    sortie: float
    flux: float
    paroi_interne: float
    paroi_externe: float


def _resistance_de_conduction(c: Cheminee) -> tuple[float, float]:
    """`(R en K/W, rayon exterieur)` des couches en serie sur toute la hauteur."""
    rayon, total = c.diametre_interieur / 2, 0.0
    for epaisseur, conductivite in c.couches:
        total += Tube(rayon, rayon + epaisseur, conductivite).resistance()
        rayon += epaisseur
    return total / c.hauteur, rayon


def etat_de_la_cheminee(c: Cheminee, k_exterieur: float) -> Etat:
    """Le bilan, `k_exterieur` etant le coefficient (convection + rayonnement)."""
    conduction, rayon = _resistance_de_conduction(c)
    s_int = math.pi * c.diametre_interieur * c.hauteur
    s_ext = 2 * math.pi * rayon * c.hauteur
    r_int = 1 / (c.coefficient_interieur * s_int)
    resistance = r_int + conduction + 1 / (k_exterieur * s_ext)
    efficacite = rendement(
        Sens.CONTRECOURANT, 1 / (c.capacite_des_fumees * resistance), 0.0
    )
    sortie = c.entree - efficacite * (c.entree - c.ambiance)
    flux = c.capacite_des_fumees * (c.entree - sortie)
    interne = (c.entree + sortie) / 2 - flux * r_int
    return Etat(sortie, flux, interne, interne - flux * conduction)


def paroi_coherente(
    c: Cheminee, k_exterieur: Callable[[float], float], *, depart: float
) -> float:
    """La paroi exterieure telle que `k_exterieur(t)` redonne `t` ; leve sinon."""
    courante = depart
    for _ in range(ITERATIONS):
        suivante = etat_de_la_cheminee(c, k_exterieur(courante)).paroi_externe
        if abs(suivante - courante) < TOLERANCE:
            return suivante
        courante = suivante
    msg = f"la paroi exterieure ne se stabilise pas apres {ITERATIONS} iterations"
    raise ValueError(msg)
