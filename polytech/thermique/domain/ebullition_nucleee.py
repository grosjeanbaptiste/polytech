"""L'ebullition nucleee : la surchauffe, le coefficient et le flux de burn-out.

Syllabus de thermique, tome 2, premiere partie, sections 6.2 a 6.6. Une
bulle de rayon `r` ne tient que si sa pression depasse celle du liquide
de `2 sigma / r`. La loi de Kelvin y ajoute l'ecart entre la tension de
vapeur d'un interface convexe et celle d'un interface plan : le liquide
doit etre surchauffe de

    t_l - t_sat = (2 sigma / r)(rho_l / (rho_l - rho_v)) dt_sat/dp,

d'autant plus que les germes sont petits.

En ebullition nucleee, Rosenhow relie la surchauffe de la paroi au flux,
par une constante `B` propre au couple liquide-surface -- de 0,003 a
0,015, 0,013 pour l'eau sur le cuivre ou l'acier. Le flux croit comme le
cube de la surchauffe, jusqu'au flux de burn-out, que Zuber tire d'une
condition d'instabilite et qui ne depend que des proprietes du fluide.

En ebullition locale, dans un liquide sous-refroidi en convection
forcee, Jens et Lottes donnent `t_p - t_sat = 2,57 phi^0,25 exp(-p/62)`,
le flux en W/m^2 et la pression en bars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from polytech.fluides.domain.grandeurs import PESANTEUR

CONSTANTES_DE_ROSENHOW: Final = (0.003, 0.015)
"""Page 86 : la plage de la constante experimentale `B`."""


@dataclass(frozen=True, slots=True)
class FluideBouillant:
    """`l_v`, `rho_l`, `rho_v`, `sigma`, et pour Rosenhow `mu_l`, `c_l`, `Pr_l`."""

    chaleur_latente: float
    liquide: float
    vapeur: float
    tension: float
    viscosite: float | None = None
    chaleur_massique: float | None = None
    prandtl: float | None = None

    def poids_apparent(self) -> float:
        """`g (rho_l - rho_v)`, en N/m^3."""
        return PESANTEUR * (self.liquide - self.vapeur)


def surpression_de_la_bulle(tension: float, rayon: float) -> float:
    """`p_b - p = 2 sigma / r`, en pascals."""
    if rayon <= 0:
        msg = f"rayon nul ou negatif : {rayon} m"
        raise ValueError(msg)
    return 2 * tension / rayon


def surchauffe_minimale(fluide: FluideBouillant, rayon: float, pente: float) -> float:
    """`t_l - t_sat`, `pente` etant `dt_sat/dp` sur la courbe de saturation."""
    kelvin = fluide.liquide / (fluide.liquide - fluide.vapeur)
    return surpression_de_la_bulle(fluide.tension, rayon) * kelvin * pente


def _exiger(propriete: float | None) -> float:
    if propriete is None:
        msg = "proprietes du liquide manquantes : viscosite, chaleur massique, Prandtl"
        raise ValueError(msg)
    return propriete


def surchauffe_de_rosenhow(
    fluide: FluideBouillant, flux: float, *, constante: float = 0.013
) -> float:
    """`c_l Dt / l_v = B [(phi / mu l_v) sqrt(sigma / g Drho)]^1/3 Pr^1,7`."""
    bas, haut = CONSTANTES_DE_ROSENHOW
    if not bas <= constante <= haut:
        msg = f"constante hors de [{bas}, {haut}] : {constante}"
        raise ValueError(msg)
    viscosite = _exiger(fluide.viscosite)
    echelle = fluide.chaleur_latente / _exiger(fluide.chaleur_massique)
    capillaire = math.sqrt(fluide.tension / fluide.poids_apparent())
    reduit = flux / (viscosite * fluide.chaleur_latente) * capillaire
    prandtl = _exiger(fluide.prandtl)
    return float(echelle * constante * reduit ** (1 / 3) * prandtl**1.7)


def flux_de_zuber(fluide: FluideBouillant) -> float:
    """`0,115 l_v rho_v [sigma g Drho / rho_v^2]^1/4 (1 + rho_v / rho_l)^1/2`."""
    instabilite = fluide.tension * fluide.poids_apparent() / fluide.vapeur**2
    correction = math.sqrt(1 + fluide.vapeur / fluide.liquide)
    moteur = 0.115 * fluide.chaleur_latente * fluide.vapeur
    return float(moteur * instabilite**0.25 * correction)


def surchauffe_de_jens_et_lottes(flux: float, pression: float) -> float:
    """`t_p - t_sat = 2,57 phi^0,25 exp(-p / 62)`, `p` en bars."""
    if flux <= 0:
        msg = f"flux nul ou negatif : {flux} W/m^2"
        raise ValueError(msg)
    return float(2.57 * flux**0.25 * math.exp(-pression / 62))
