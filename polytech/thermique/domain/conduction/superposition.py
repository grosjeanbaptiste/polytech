"""Le principe de superposition en conduction variable.

Syllabus de thermique, tome 1, premiere partie, section 4.4. L'equation
de la chaleur est lineaire : la somme de deux solutions en est une, pour
la somme de leurs conditions initiales et aux limites. Un probleme
complique se decompose donc en echelons, dont la reponse `F_1(tau)` --
la fraction de l'ecart deja comblee -- est connue.

La trempe etagee en est le premier usage : un second bain, applique a
l'instant `tau_1`, ajoute sa reponse decalee. A la limite d'echelons de
plus en plus fins, un fluide dont la temperature varie continument donne
l'integrale de Duhamel,

`t(tau) = t_0 + (t_f(0) - t_0) F_1(tau)
          + integrale de (dt_f/dtau*) F_1(tau - tau*) dtau*`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.integrate import quad

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type Reponse = Callable[[float], float]
"""`F_1(tau)`, nulle avant l'instant zero."""


def trempe_etagee(
    reponse: Reponse,
    duree: float,
    *,
    initiale: float,
    bains: Sequence[tuple[float, float]],
) -> float:
    """La temperature a l'instant `duree`, les bains etant (instant, temperature)."""
    temperature, precedente = initiale, initiale
    for instant, fluide in bains:
        if duree > instant:
            temperature += (fluide - precedente) * reponse(duree - instant)
        precedente = fluide
    return temperature


def integrale_de_duhamel(
    reponse: Reponse,
    derivee_du_fluide: Callable[[float], float],
    duree: float,
    *,
    initiale: float,
    saut: float = 0.0,
) -> float:
    """`t(tau)` pour un fluide variable, apres un saut initial eventuel."""
    integrale, _ = quad(
        lambda instant: derivee_du_fluide(instant) * reponse(duree - instant),
        0.0,
        duree,
    )
    return initiale + saut * reponse(duree) + float(integrale)
