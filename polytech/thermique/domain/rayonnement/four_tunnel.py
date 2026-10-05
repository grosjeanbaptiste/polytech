"""Les elements de calcul d'un four tunnel electrothermique.

Syllabus de thermique, tome 1, deuxieme partie, section 3.10. Des toles
pendues au milieu du four sont chauffees sur leurs deux faces par des
resistances adossees aux parois laterales. Les resistances et leur
paroi sont remplacees par un plan gris fictif (1), de hauteur `L_1` ;
la charge (2), de hauteur `L_2`, n'occupe que la fraction `beta` de la
longueur ; le reste de l'enceinte (R) est adiabatique.

Par metre de four, la regle des elastiques donne les aires d'echange
direct, et le schema a trois zones l'aire d'echange total. Le bilan
d'une tranche, `S_1S_2 sigma (T_1^4 - T_2^4) dx = q_m c dT_2`, s'integre
alors en la longueur qu'il faut pour porter la charge a sa temperature
de sortie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.thermique.domain.rayonnement.corps_noir import STEFAN_BOLTZMANN


@dataclass(frozen=True, slots=True)
class FourTunnel:
    """Les hauteurs `L_1` et `L_2`, la demi-largeur `L_4`, `beta`, `(eps_1, eps_2)`."""

    source: float
    charge: float
    demi_largeur: float
    remplissage: float
    emissivites: tuple[float, float]

    def __post_init__(self) -> None:
        """Refuse `L_2 < L_1` et un remplissage hors de ]0, 1]."""
        if self.charge < self.source:
            msg = f"charge plus courte que la source : {self.charge} m"
            raise ValueError(msg)
        if not 0 < self.remplissage <= 1:
            msg = f"remplissage hors de ]0, 1] : {self.remplissage}"
            raise ValueError(msg)

    def aires_directes(self) -> dict[str, float]:
        """`12`, `11`, `1R` et `2R`, en m^2 par metre de four."""
        largeur = self.demi_largeur
        croisees = math.hypot(self.source, largeur) + math.hypot(self.charge, largeur)
        droites = largeur + math.hypot(largeur, self.charge - self.source)
        vers_la_charge = self.remplissage * (croisees - droites)
        en_face = math.hypot(self.source, 2 * largeur) - 2 * largeur
        entre_sources = 2 * (1 - self.remplissage) * en_face
        return {
            "12": vers_la_charge,
            "11": entre_sources,
            "1R": 2 * self.source - entre_sources - vers_la_charge,
            "2R": 2 * self.charge * self.remplissage - vers_la_charge,
        }

    def aire_d_echange_total(self) -> float:
        """`S_1S_2|R`, en m^2 par metre de four."""
        directes = self.aires_directes()
        premiere, seconde = self.emissivites
        source = (1 - premiere) / (2 * premiere * self.source)
        charge = (1 - seconde) / (2 * seconde * self.charge * self.remplissage)
        relais = 1 / (1 / directes["1R"] + 1 / directes["2R"])
        return 1 / (source + charge + 1 / (directes["12"] + relais))

    def longueur(
        self, debit: float, source: float, *, entree: float, sortie: float
    ) -> float:
        """`L`, pour un debit de capacite `q_m c` en W/K et des kelvins."""
        if sortie >= source:
            msg = f"la charge ne depasse pas la source : {sortie} K"
            raise ValueError(msg)

        def primitive(charge: float) -> float:
            logarithme = math.log((source + charge) / (source - charge))
            return logarithme + 2 * math.atan(charge / source)

        echange = 4 * source**3 * STEFAN_BOLTZMANN * self.aire_d_echange_total()
        return debit * (primitive(sortie) - primitive(entree)) / echange
