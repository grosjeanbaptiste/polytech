"""L'isolation par une lame d'air : la vitre simple et le double vitrage.

Syllabus de thermique, tome 2, deuxieme partie, section 1.7. Une vitre
seule ne vaut presque rien : sa resistance, `a / lambda_v`, est dix fois
plus petite que celle des deux films d'air qui la baignent. Enfermer une
lame d'air entre deux vitres ajoute une vraie resistance en serie :

    1/K_gl = 1/K_e + a/lambda_v + 1/(K_c + K_r) + a/lambda_v + 1/K_i.

Dans la lame, deux echanges agissent en parallele. La conductibilite,
`K_c = lambda_eq / b`, que la convection naturelle renforce des que
`Gr_b Pr` depasse 1400. Et le rayonnement entre les deux vitres, que le
verre, opaque au-dela de 4 micrometres, se renvoie comme deux plans
gris : `K_r = 4 sigma T_m^3 / (2 / epsilon - 1)`.

Les temperatures des deux vitres ne sont pas connues d'avance ; l'ecart
entre elles resulte de la continuite du flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polytech.thermique.domain.continuite_du_flux import temperature_superficielle
from polytech.thermique.domain.convection.naturelle import nusselt_lame_de_fluide
from polytech.thermique.domain.rayonnement_equivalent import coefficient_sigma

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def coefficient_de_la_lame(
    epaisseur: float, conductivite: float, rayleigh: float
) -> float:
    """`K_c = lambda_eq / b`, ou `lambda_eq / lambda_a` est le Nusselt de la lame."""
    if epaisseur <= 0:
        msg = f"epaisseur de lame nulle ou negative : {epaisseur} m"
        raise ValueError(msg)
    return conductivite * nusselt_lame_de_fluide(rayleigh) / epaisseur


def coefficient_de_rayonnement_entre_vitres(emissivite: float, moyenne: float) -> float:
    """`K_r = 4 sigma T_m^3 / (2 / epsilon - 1)`, la relation linearisee."""
    if not 0 < emissivite <= 1:
        msg = f"facteur d'emission hors de ]0, 1] : {emissivite}"
        raise ValueError(msg)
    return coefficient_sigma(moyenne, moyenne) / (2 / emissivite - 1)


@dataclass(frozen=True, slots=True)
class Vitrage:
    """Des vitres identiques, et les coefficients `(K_e, K_i)` de leurs deux faces."""

    epaisseur: float
    conductivite: float
    echanges: tuple[float, float]

    def __post_init__(self) -> None:
        """Refuse un coefficient conventionnel nul."""
        for echange in self.echanges:
            if echange <= 0:
                msg = f"coefficient de transmission nul ou negatif : {echange}"
                raise ValueError(msg)

    def resistance_hors_des_lames(self, vitres: int) -> float:
        """Les deux films et `vitres` epaisseurs de verre, en m^2 K/W."""
        films = sum(1 / echange for echange in self.echanges)
        return films + vitres * self.epaisseur / self.conductivite

    def coefficient_global(self, lames: Sequence[float]) -> float:
        """`K_gl` pour autant de lames, donc une vitre de plus ; `()` : vitre simple."""
        for lame in lames:
            if lame <= 0:
                msg = f"coefficient de lame nul ou negatif : {lame}"
                raise ValueError(msg)
        enfermees = sum(1 / lame for lame in lames)
        return 1 / (self.resistance_hors_des_lames(len(lames) + 1) + enfermees)


def ecart_entre_les_vitres(
    vitrage: Vitrage, ecart: float, lame: Callable[[float], float]
) -> float:
    """`t_1 - t_2` d'un double vitrage ; `lame` rend `K_c + K_r` pour cet ecart."""
    externe = vitrage.resistance_hors_des_lames(2)

    def entrant(entre_vitres: float) -> float:
        return (ecart - entre_vitres) / externe

    def sortant(entre_vitres: float) -> float:
        return lame(entre_vitres) * entre_vitres

    return temperature_superficielle(entrant, sortant, (0.0, ecart))
