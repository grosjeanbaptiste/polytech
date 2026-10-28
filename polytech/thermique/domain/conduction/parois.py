"""La conduction stationnaire dans une plaque et dans un tube.

Syllabus de thermique, tome 1, premiere partie, sections 2.1 et 2.2.
Sans generation et en regime permanent, le champ est laplacien.

Dans une plaque le profil est lineaire, `phi = (lambda / e)(t_1 - t_2)`,
et `e / lambda` est une resistance thermique par unite de surface : des
plaques accolees, puis les deux fluides qui les baignent, sont autant de
resistances en serie.

Dans un tube le profil est logarithmique, et c'est le flux par unite de
longueur qui se conserve, `Phi = 2 pi lambda (t_1 - t_2) / ln(r_2 / r_1)`.
La resistance lineique vaut `ln(r_2 / r_1) / (2 pi lambda)` ; un fluide
y ajoute `1 / (2 pi r K)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import sympy as sp

from polytech.thermique.domain.reseaux import en_serie

if TYPE_CHECKING:
    from collections.abc import Sequence

type Echange = tuple[float, float]
"""Le coefficient de transmission d'un fluide, et sa temperature."""


@dataclass(frozen=True, slots=True)
class Couche:
    """Une plaque : son epaisseur et sa conductivite."""

    epaisseur: float
    conductivite: float

    def resistance(self) -> float:
        """`e / lambda`, par unite de surface."""
        return self.epaisseur / self.conductivite


@dataclass(frozen=True, slots=True)
class Tube:
    """Un tube : ses deux rayons et sa conductivite."""

    rayon_interieur: float
    rayon_exterieur: float
    conductivite: float

    def __post_init__(self) -> None:
        """Refuse un tube sans epaisseur."""
        if self.rayon_exterieur <= self.rayon_interieur:
            msg = f"rayon exterieur non superieur a l'interieur : {self}"
            raise ValueError(msg)

    def resistance(self) -> float:
        """`ln(r_2 / r_1) / (2 pi lambda)`, par unite de longueur."""
        rapport = self.rayon_exterieur / self.rayon_interieur
        return math.log(rapport) / (2 * math.pi * self.conductivite)


def profil_de_plaque(
    abscisse: sp.Expr, epaisseur: sp.Expr, chaude: sp.Expr, froide: sp.Expr
) -> sp.Expr:
    """`t = t_1 + (t_2 - t_1) x / e`."""
    return chaude + (froide - chaude) * abscisse / epaisseur


def flux_de_plaque(
    conductivite: float, epaisseur: float, chaude: float, froide: float
) -> float:
    """`phi = (lambda / e)(t_1 - t_2)`, en W/m^2."""
    return (chaude - froide) / Couche(epaisseur, conductivite).resistance()


def temperatures_d_interface(
    couches: Sequence[Couche], chaude: float, froide: float
) -> tuple[list[float], float]:
    """Les temperatures entre les couches, et la densite de flux."""
    flux = (chaude - froide) / en_serie(*(c.resistance() for c in couches))
    interfaces, courante = [], chaude
    for couche in couches[:-1]:
        courante -= flux * couche.resistance()
        interfaces.append(courante)
    return (interfaces, flux)


def flux_entre_deux_fluides(
    couches: Sequence[Couche], *, interieur: Echange, exterieur: Echange
) -> float:
    """`phi = (t_i - t_e) / (1 / K_i + somme(e / lambda) + 1 / K_e)`."""
    if interieur[0] <= 0 or exterieur[0] <= 0:
        msg = "coefficient de transmission nul ou negatif"
        raise ValueError(msg)
    parois = (couche.resistance() for couche in couches)
    totale = en_serie(1 / interieur[0], *parois, 1 / exterieur[0])
    return (interieur[1] - exterieur[1]) / totale


def profil_de_tube(
    rayon: sp.Expr, rayons: tuple[sp.Expr, sp.Expr], faces: tuple[sp.Expr, sp.Expr]
) -> sp.Expr:
    """`t = t_1 + (t_2 - t_1) ln(r / r_1) / ln(r_2 / r_1)`."""
    reduit = sp.log(rayon / rayons[0]) / sp.log(sp.S(rayons[1]) / rayons[0])
    return faces[0] + (faces[1] - faces[0]) * reduit


def flux_lineique_de_tubes(
    tubes: Sequence[Tube],
    chaude: float,
    froide: float,
    *,
    interieur: float | None = None,
    exterieur: float | None = None,
) -> float:
    """`Phi` par metre, a travers des tubes coaxiaux et, s'il y en a, deux fluides."""
    resistances = [tube.resistance() for tube in tubes]
    if interieur is not None:
        rayon = tubes[0].rayon_interieur
        resistances.append(1 / (2 * math.pi * rayon * interieur))
    if exterieur is not None:
        rayon = tubes[-1].rayon_exterieur
        resistances.append(1 / (2 * math.pi * rayon * exterieur))
    return (chaude - froide) / en_serie(*resistances)


def densite_de_flux_a_la_face(flux_lineique: float, rayon: float) -> float:
    """`phi = Phi / (2 pi r)`, en W/m^2, sur la face de rayon `rayon`."""
    return flux_lineique / (2 * math.pi * rayon)


def conductivite_interpolee(
    table: Sequence[tuple[float, float]], temperature: float
) -> float:
    """La conductivite a cette temperature, par interpolation lineaire."""
    for (basse, avant), (haute, apres) in pairwise(table):
        if basse <= temperature <= haute:
            part = (temperature - basse) / (haute - basse)
            return avant + part * (apres - avant)
    msg = f"temperature hors de la table : {temperature}"
    raise ValueError(msg)
