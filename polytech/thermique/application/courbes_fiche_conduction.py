"""Les courbes de conduction de la fiche de thermique, calculees par la bibliotheque.

Chaque fonction rend les points `(x, y)` d'une courbe de la fiche ; la fiche
ne porte plus que des noms (`@courbe:nom@`), jamais un nombre tape.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_exact,
)
from polytech.thermique.domain.conduction.ailettes import AiletteDroite
from polytech.thermique.domain.conduction.barre_isolee import profil_de_barre_isolee
from polytech.thermique.domain.conduction.parois import Tube, flux_lineique_de_tubes
from polytech.thermique.domain.conduction.rayon_critique import rayon_critique
from polytech.thermique.domain.continuite_du_flux import temperature_superficielle
from polytech.thermique.domain.inertie import SolideIsotherme

type Points = list[tuple[float, float]]

CONVECTION: Final = 10.0
"""`K_c` du toit, en W/(m^2 K) : un exemple, la fiche n'en donne pas."""
ABSORBE: Final = 290.0
"""Flux solaire absorbe par le toit, en W/m^2 : un exemple."""
AMBIANT: Final = 300.0
EMISSIVITE: Final = 0.9
BORNES_DU_BILAN: Final = (280.0, 360.0)
CORDE: Final = (300.0, 340.0)
RAPPORT_MAXIMAL: Final = 8.0


def ecart_de_bilan(temperature: float) -> float:
    """Absorbe moins ce que le toit perd (convection plus rayonnement)."""
    perte = CONVECTION + coefficient_de_rayonnement_exact(
        EMISSIVITE, temperature, AMBIANT
    )
    return ABSORBE - perte * (temperature - AMBIANT)


def bilan_du_toit() -> Points:
    """`entrant - sortant` en fonction de la temperature posee."""
    bas, haut = BORNES_DU_BILAN
    return [(t, ecart_de_bilan(t)) for t in range(int(bas), int(haut) + 1, 4)]


def cordes() -> Points:
    """Les deux points de la corde de la fausse position."""
    return [(t, ecart_de_bilan(t)) for t in CORDE]


def racine_de_la_corde() -> float:
    """Ou la corde coupe l'axe : la premiere estimation de la fausse position."""
    (t1, g1), (t2, g2) = cordes()
    return t1 - g1 * (t2 - t1) / (g2 - g1)


def racine_du_bilan() -> float:
    """La temperature qui annule le bilan, par la fausse position de la bibliotheque."""
    perte = lambda t: ABSORBE - ecart_de_bilan(t)  # noqa: E731 -- fermeture d'une ligne
    return temperature_superficielle(lambda _: ABSORBE, perte, BORNES_DU_BILAN)


def flux_du_tube_isole() -> Points:
    """Flux par metre / `2 pi` selon `r_e / r_i` (`lambda = 2 K r_i`, `Dt = 1`)."""
    pas = [1.02 + 0.07 * i for i in range(100) if 1.02 + 0.07 * i <= RAPPORT_MAXIMAL]
    return [
        (
            x,
            flux_lineique_de_tubes([Tube(1.0, x, 2.0)], 1.0, 0.0, exterieur=1.0)
            / (2 * math.pi),
        )
        for x in pas
    ]


def rayon_critique_reduit() -> float:
    """`r_c / r_i` pour `lambda = 2`, `K = 1`, `r_i = 1`."""
    return rayon_critique(2.0, 1.0)


def flux_au_rayon_critique() -> float:
    """Le flux relatif au maximum, la ou le rayon vaut `r_c`."""
    r = rayon_critique_reduit()
    return flux_lineique_de_tubes([Tube(1.0, r, 2.0)], 1.0, 0.0, exterieur=1.0) / (
        2 * math.pi
    )


def barre_refroidie() -> Points:
    """`T - T_p` reduit, de `-1` a `1` : la demi-plaque jusqu'a son plan isole."""
    x = sp.Symbol("x", nonnegative=True)
    profil = profil_de_barre_isolee(x, 1, 0, 1)
    centre = float(profil.subs(x, 1))
    points = [i / 10 - 1 for i in range(21)]
    return [(p, float(profil.subs(x, 1 - abs(p))) / centre) for p in points]


def decroissance_du_solide() -> Points:
    """`exp(-t / tau_c)` pour un solide isotherme, `t / tau_c` de 0 a 5."""
    solide = SolideIsotherme(resistance=1.0, capacite=1.0)
    return [(i / 10, float(solide.ecart_reduit(i / 10))) for i in range(51)]


def tangente_du_solide() -> Points:
    """La tangente a l'origine, tiree de la derivee de la decroissance."""
    tau = sp.Symbol("tau", nonnegative=True)
    pente = float(
        sp.diff(SolideIsotherme(1.0, 1.0).ecart_reduit(tau), tau).subs(tau, 0)
    )
    return [(0.0, 1.0), (1.0, 1.0 + pente)]


def efficacite_d_ailette() -> Points:
    """`tanh(m l) / (m l)` pour `m l` de 0,01 a 3 (`m = 1`, `l` variable)."""
    return [
        (x, AiletteDroite(1.0, x).efficacite())
        for x in (0.01 + 2.99 * i / 99 for i in range(100))
    ]
