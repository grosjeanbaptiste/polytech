"""Deux conducteurs cylindriques paralleles, par la methode des images.

Le syllabus d'electricite generale (exemple 3.8.2, suite) materialise deux
equipotentielles du champ de deux droites opposees. Pour deux cylindres de
rayons `R_1` et `R_2` dont les axes sont distants de `2b` :

    x_01 = +-[(R_2^2 - R_1^2) / 4b - b]                     (3.8.17)
    x_02 = +-[(R_2^2 - R_1^2) / 4b + b]                     (3.8.17)
    a^2  = x_01^2 - R_1^2 = x_02^2 - R_2^2                  (3.8.16)
    p^2  = (x_0 - a) / (x_0 + a)                            (3.8.19)
    C / l = 2 pi eps_0 / ln(p_1 / p_2)                      (3.8.20)

le signe superieur valant pour deux cylindres exterieurs l'un a l'autre,
le signe inferieur quand l'un entoure l'autre. Le champ est maximal sur le
plus petit cylindre, face a l'autre (3.8.22).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE


@dataclass(frozen=True, slots=True)
class DeuxCylindres:
    """Deux cylindres paralleles : leurs rayons et le demi-entraxe `b`."""

    premier_rayon: float
    second_rayon: float
    demi_entraxe: float

    def __post_init__(self) -> None:
        """Refuse deux cylindres qui se coupent."""
        entraxe = 2 * self.demi_entraxe
        somme = self.premier_rayon + self.second_rayon
        difference = abs(self.premier_rayon - self.second_rayon)
        if difference <= entraxe <= somme:
            bornes = f"entre {difference} et {somme}"
            msg = f"les deux cylindres se coupent : entraxe {entraxe} {bornes}"
            raise ValueError(msg)

    def _signe(self) -> int:
        exterieurs = 2 * self.demi_entraxe > self.premier_rayon + self.second_rayon
        return 1 if exterieurs else -1

    def axes(self) -> tuple[float, float]:
        """`(x_01, x_02)` : les abscisses des deux axes, l'origine entre les droites."""
        decalage = (self.second_rayon**2 - self.premier_rayon**2) / (
            4 * self.demi_entraxe
        )
        signe = self._signe()
        return (
            signe * (decalage - self.demi_entraxe),
            signe * (decalage + self.demi_entraxe),
        )

    def demi_ecart_des_droites(self) -> float:
        """`a` : la demi-distance des deux droites equivalentes."""
        premier_axe, _ = self.axes()
        return math.sqrt(premier_axe**2 - self.premier_rayon**2)

    def parametres(self) -> tuple[float, float]:
        """`(p_1, p_2)` : les parametres des deux equipotentielles materialisees."""
        a = self.demi_ecart_des_droites()
        premier, second = (math.sqrt((x - a) / (x + a)) for x in self.axes())
        return (premier, second)

    def capacite_lineique(self) -> float:
        """`C / l = 2 pi eps_0 / ln(p_1 / p_2)`, en farads par metre."""
        p_1, p_2 = self.parametres()
        return 2 * math.pi * PERMITTIVITE_DU_VIDE / math.log(p_1 / p_2)

    def champ_maximal(self, tension: float) -> float:
        """Le champ sur le plus petit cylindre, la ou il fait face a l'autre."""
        p_1, p_2 = self.parametres()
        petit_en_premier = self.premier_rayon <= self.second_rayon
        p, rayon = (
            (p_1, self.premier_rayon) if petit_en_premier else (p_2, self.second_rayon)
        )
        return (p + 1) / abs(p - 1) * tension / (rayon * math.log(p_1 / p_2))


def capacite_lineique_de_deux_cylindres_egaux(
    rayon: float, demi_entraxe: float
) -> float:
    """`C / l = pi eps_0 / ln[b/R + sqrt((b/R)^2 - 1)]` (3.8.23)."""
    rapport = demi_entraxe / rayon
    return (
        math.pi * PERMITTIVITE_DU_VIDE / math.log(rapport + math.sqrt(rapport**2 - 1))
    )


def capacite_lineique_des_petits_rayons(
    premier: float, second: float, demi_entraxe: float
) -> float:
    """`C / l ~ 2 pi eps_0 / ln(4 b^2 / R_1 R_2)`, pour des fils fins (3.8.24)."""
    return (
        2
        * math.pi
        * PERMITTIVITE_DU_VIDE
        / math.log(4 * demi_entraxe**2 / (premier * second))
    )


def capacite_lineique_presque_coaxiale(
    grand: float, petit: float, demi_entraxe: float
) -> float:
    """`C / l ~ 2 pi eps_0 / ln[(R_1/R_2) (2a - b)/(2a + b)]` (3.8.25 et 3.8.26)."""
    a = (grand**2 - petit**2) / (4 * demi_entraxe)
    correction = (2 * a - demi_entraxe) / (2 * a + demi_entraxe)
    return 2 * math.pi * PERMITTIVITE_DU_VIDE / math.log(grand / petit * correction)
