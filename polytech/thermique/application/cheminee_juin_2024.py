"""La cheminee d'usine isolee (exercice cote de juin 2024, question bonus).

Source : `cours-markdown/ThermiqueFondements/EXAMENS/`
`Exam_2022_2025_Exercices_cotes_resolus.md`. Fumees : air a 150 C
(`air_sec`), 5 kg/s ; air du vent a -10 C : `nu = 1,2430e-5`, `Pr = 0,716`,
`lambda = 0,0233`. Le corrige applique la derniere plage du formulaire du
cylindre (`Re = 4,18e5`, juste au-dela de `4e5`) : la meme ici, signalee.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.thermique.domain.cheminee import Cheminee
from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_exact,
)
from polytech.thermique.domain.convection.forcee import TABLE_DU_CYLINDRE
from polytech.thermique.domain.grandeurs import ZERO_ABSOLU
from polytech.thermique.domain.tables.air_sec import air_sec

VENT: Final = 5.0
DIAMETRE_EXTERIEUR_NU: Final = 1.04
AIR: Final = (1.2430e-5, 0.716, 0.0233)
"""`(nu, Pr, lambda)` de l'air a -10 C."""
EMISSIVITE: Final = 0.6
KELVIN: Final = ZERO_ABSOLU
DEBIT: Final = 5.0
FUMEES: Final = air_sec(150.0 + KELVIN)


def coefficient_interieur() -> tuple[float, float]:
    """`(K, Nu)` de l'ecoulement turbulent : `Re = 4 q / (pi D mu)`, `D = 1 m`."""
    reynolds = 4 * DEBIT / (math.pi * 1.0 * FUMEES.viscosite)
    nusselt = 0.0234 * reynolds**0.8 * FUMEES.prandtl ** (1 / 3)
    return nusselt * FUMEES.conductivite / 1.0, nusselt


def _convection_du_vent() -> float:
    nu, prandtl, conductivite = AIR
    reynolds = VENT * DIAMETRE_EXTERIEUR_NU / nu
    _, _, coefficient, exposant = TABLE_DU_CYLINDRE[-1]
    nusselt = coefficient * reynolds**exposant * prandtl ** (1 / 3)
    return float(nusselt * conductivite / DIAMETRE_EXTERIEUR_NU)


def coefficient_exterieur(paroi_celsius: float) -> float:
    """Vent plus rayonnement vers l'ambiance a -10 C, la paroi etant supposee."""
    rayonnement = coefficient_de_rayonnement_exact(
        EMISSIVITE, paroi_celsius + KELVIN, -10.0 + KELVIN
    )
    return _convection_du_vent() + rayonnement


CHEMINEE: Final = Cheminee(
    hauteur=30.0,
    diametre_interieur=1.0,
    couches=((0.02, 15.0), (0.05, 0.04)),
    capacite_des_fumees=DEBIT * FUMEES.chaleur_massique,
    entree=150.0,
    ambiance=-10.0,
    coefficient_interieur=coefficient_interieur()[0],
)
