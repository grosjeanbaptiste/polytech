"""Les figures de la fiche de thermique dont les courbes portent un nombre.

Les figures sont dessinees a la main dans `cursus/fiches/ThermiqueFondements.toml` :
ici, les trois dont l'ordonnee se recalcule par la bibliotheque.
"""

from __future__ import annotations

import math

from polytech.maths.domain.comparaison import proches_relativement
from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_linearise,
)
from polytech.thermique.domain.rayonnement.corps_noir import (
    emittance,
    exitance_spectrale,
    longueur_d_onde_du_maximum,
)
from polytech.thermique.domain.rayonnement.lambert import SECONDE_CONSTANTE
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    flux_entre_deux_plans_gris,
)

MICRO = 1e-6


def test_la_courbe_de_planck_normalisee_de_la_figure_du_corps_noir() -> None:
    """Courbe `(l_m/l)^5 (e^(c_2/l_m) - 1)/(e^(c_2/l) - 1)`, `c_2 = 14388 um K`."""
    assert proches_relativement(SECONDE_CONSTANTE / MICRO, 14388.0, 1e-4)
    for temperature in (5800.0, 310.0):
        maximum = longueur_d_onde_du_maximum(temperature)
        for microns in (0.3, 1.0, 5.0, 12.0):
            lam = microns * MICRO
            figure = ((2898 / temperature / microns) ** 5) * (
                (math.exp(14388 / 2898) - 1)
                / (math.exp(14388 / (microns * temperature)) - 1)
            )
            obtenu = exitance_spectrale(lam, temperature) / exitance_spectrale(
                maximum, temperature
            )
            assert proches_relativement(obtenu, figure, 1e-2)


def test_la_courbe_du_coefficient_de_rayonnement_de_la_figure() -> None:
    """`h_r = 4 sigma T^3` pour `epsilon = 1` (le trace arrondit `sigma` a 5,67e-8)."""
    for t in (250.0, 300.0, 400.0, 600.0):
        obtenu = coefficient_de_rayonnement_linearise(1.0, t, t)
        assert proches_relativement(obtenu, 4 * 5.67e-8 * t**3, 1e-3)


def test_la_courbe_de_la_feuille_d_aluminium_de_la_figure() -> None:
    """`F = 1/(1/0.9 + 1/x - 1)` : le facteur de deux plans gris, `epsilon_1 = 0,9`."""
    t1, t2 = 400.0, 300.0
    for eps2 in (0.05, 0.3, 0.9):
        flux = flux_entre_deux_plans_gris(0.9, eps2, t1, t2, 1.0)
        facteur = flux / (emittance(t1) - emittance(t2))
        assert proches_relativement(facteur, 1 / (1 / 0.9 + 1 / eps2 - 1), 1e-12)
