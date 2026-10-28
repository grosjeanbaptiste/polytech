"""Les courbes de rayonnement de la fiche de thermique, par la bibliotheque."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_linearise,
)
from polytech.thermique.domain.rayonnement.corps_noir import (
    emittance,
    exitance_spectrale,
    longueur_d_onde_du_maximum,
)
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    flux_entre_deux_plans_gris,
)
from polytech.thermique.domain.rayonnement.vitre_selective import VitreSelective

if TYPE_CHECKING:
    from polytech.thermique.application.courbes_fiche_conduction import Points

MICRO: Final = 1e-6
SOLEIL: Final = 5800.0
CORPS: Final = 310.0
"""Les deux temperatures de la fiche (le soleil, le corps humain), en kelvins."""
VERRE: Final = VitreSelective(0.94, 0.06, 3.5e-6)
"""Exercice de la serre (juin 2025) : `tau = 0,94` sous 3,5 um, nul au-dela."""
EMISSIVITE_DU_PLAN: Final = 0.9


def _logarithmique(debut: float, fin: float, nombre: int) -> list[float]:
    raison = (fin / debut) ** (1 / (nombre - 1))
    return [debut * raison**i for i in range(nombre)]


def planck_reduit(temperature: float) -> Points:
    """`M_lambda / M_lambda,max`, `lambda` de 0,2 a 50 um."""
    maximum = exitance_spectrale(longueur_d_onde_du_maximum(temperature), temperature)
    return [
        (lam, exitance_spectrale(lam * MICRO, temperature) / maximum)
        for lam in _logarithmique(0.2, 50.0, 90)
    ]


def planck_du_soleil() -> Points:
    """La courbe de la fiche a 5800 K."""
    return planck_reduit(SOLEIL)


def planck_du_corps() -> Points:
    """La courbe de la fiche a 310 K."""
    return planck_reduit(CORPS)


def transmission_du_verre() -> Points:
    """`tau_lambda` en marches : les points ou elle change (trace `const plot`)."""
    bornes = (0.2, VERRE.coupure / MICRO, 50.0)
    return [(lam, VERRE.transmission_spectrale(lam * MICRO)) for lam in bornes]


def facteur_de_deux_plans() -> Points:
    """`F_12` selon `epsilon_2` (`epsilon_1 = 0,9`) : flux sur `sigma dT4`."""
    t1, t2 = 400.0, 300.0
    ecart = emittance(t1) - emittance(t2)
    return [
        (e, flux_entre_deux_plans_gris(EMISSIVITE_DU_PLAN, e, t1, t2, 1.0) / ecart)
        for e in (0.02 + 0.02 * i for i in range(50))
    ]


def coefficient_de_rayonnement() -> Points:
    """`h_r = 4 sigma T^3` pour `epsilon = 1`, de 250 a 600 K."""
    return [
        (t, coefficient_de_rayonnement_linearise(1.0, t, t))
        for t in (250.0 + 350.0 * i / 59 for i in range(60))
    ]


def temperature_du_soleil() -> float:
    """Pour la legende."""
    return SOLEIL


def temperature_du_corps() -> float:
    """Pour la legende."""
    return CORPS
