"""La convection naturelle dans un chenal, et le reseau d'ailettes verticales.

Syllabus de thermique, tome 2, premiere partie, section 4.2.1 et
exercice III.4.4. Entre deux plaques verticales paralleles, distantes de
`b` et hautes de `H`, les deux couches limites se genent. Elenbaas donne

    Nu_b = K b / lambda = (q / 24) {1 - exp[-24 (0,5 / q)^0,75]},

avec `q = Gr_b Pr b / H`. Pour un chenal etroit le fluide prend la
temperature des parois et `Nu = q / 24` ; pour un chenal large les
plaques s'ignorent, et `Nu` croit comme `q^1/4`.

Un reseau d'ailettes verticales echange d'autant plus qu'elles sont
nombreuses, d'autant moins qu'elles sont serrees. Le flux varie comme
`F(q) q^-1/2`, maximum pour `q` voisin de 50 : c'est l'ecartement
optimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.fluides.domain.grandeurs import PESANTEUR
from polytech.thermique.domain.conduction.ailette_droite import (
    parametre_d_ailette_mince,
)

GROUPE_OPTIMAL = 50.0
"""Page 62 : le maximum de `F(q) q^-1/2`."""


def groupe_du_chenal(
    grashof: float, prandtl: float, *, ecartement: float, hauteur: float
) -> float:
    """`q = Gr_b Pr b / H`."""
    return grashof * prandtl * ecartement / hauteur


def nusselt_d_elenbaas(groupe: float) -> float:
    """`Nu_b = F(q)`."""
    if groupe <= 0:
        msg = f"groupe nul ou negatif : {groupe}"
        raise ValueError(msg)
    return groupe / 24 * -math.expm1(-24 * (0.5 / groupe) ** 0.75)


def ecartement_optimal(
    viscosite: float, *, hauteur: float, dilatation: float, ecart: float, prandtl: float
) -> float:
    """`b_opt = [50 nu^2 H / (beta g theta_o Pr)]^0,25`."""
    poussee = dilatation * PESANTEUR * ecart * prandtl
    return float((GROUPE_OPTIMAL * viscosite**2 * hauteur / poussee) ** 0.25)


@dataclass(frozen=True, slots=True)
class ReseauDAilettes:
    """La largeur `A` et la hauteur `H` de la plaque, la longueur `L`, `e`, `b`."""

    largeur: float
    hauteur: float
    longueur: float
    epaisseur: float
    ecartement: float

    def __post_init__(self) -> None:
        """Refuse un ecartement nul : les ailettes se toucheraient."""
        if self.ecartement <= 0:
            msg = f"ecartement nul ou negatif : {self.ecartement} m"
            raise ValueError(msg)

    def _pas(self) -> float:
        return self.ecartement + self.epaisseur

    def nombre(self) -> int:
        """`INT(A / (b + e))`."""
        return int(self.largeur / self._pas())

    def surface_des_ailettes(self) -> float:
        """`S_a = 2 L H A / (b + e)`."""
        return 2 * self.longueur * self.hauteur * self.largeur / self._pas()

    def surface_primaire(self) -> float:
        """`S_o = b A H / (b + e)`."""
        return self.ecartement * self.largeur * self.hauteur / self._pas()

    def efficacite(self, echange: float, conductivite: float) -> float:
        """`Omega = tanh(m L) / (m L)`, avec `m = sqrt(2 K / (lambda e))`."""
        reduit = parametre_d_ailette_mince(echange, conductivite, self.epaisseur)
        reduit *= self.longueur
        return math.tanh(reduit) / reduit

    def flux_des_ailettes(
        self, echange: float, ecart: float, conductivite: float
    ) -> float:
        """`2 n K theta_o L H Omega`, la surface primaire negligee."""
        surface = 2 * self.nombre() * self.longueur * self.hauteur
        return echange * ecart * surface * self.efficacite(echange, conductivite)

    def volume_de_metal(self) -> float:
        """`n e H L`, en m^3."""
        return self.nombre() * self.epaisseur * self.hauteur * self.longueur
