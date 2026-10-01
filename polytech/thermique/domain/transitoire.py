"""La conduction en regime transitoire, et le nombre qui decide de la methode.

Le nombre de Biot `Bi = K L / lambda` compare la resistance interne du
corps a sa resistance de surface. C'est lui qui autorise -- ou interdit --
le modele du corps **thermiquement mince**, ou la temperature est uniforme
et ne depend que du temps. Le formulaire ne donne d'abaques que pour le
cas contraire, precisement parce que le cas mince se traite a la main.

La garde sur `Bi < 0,1` n'est pas decorative : au-dela, le corps a un
gradient interne, la temperature uniforme est une fiction, et la constante
de temps calculee ici serait fausse sans le paraitre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polytech.thermique.domain.validation import exiger_strictement_positif

BIOT_MINCE = 0.1
"""Au-dela, le corps n'est plus thermiquement mince."""


@dataclass(frozen=True, slots=True)
class CorpsTransitoire:
    """Un corps qui se refroidit, decrit par ses grandeurs globales."""

    echange: float
    longueur_caracteristique: float
    conductivite: float
    capacite_volumique: float

    def __post_init__(self) -> None:
        """Valide les invariants a la construction."""
        for valeur, nom in (
            (self.echange, "coefficient d'echange"),
            (self.longueur_caracteristique, "longueur caracteristique"),
            (self.conductivite, "conductivite"),
            (self.capacite_volumique, "capacite volumique"),
        ):
            exiger_strictement_positif(valeur, nom)

    def biot(self) -> float:
        """`K L / lambda` : resistance interne sur resistance de surface."""
        return self.echange * self.longueur_caracteristique / self.conductivite

    def est_thermiquement_mince(self) -> bool:
        """Vrai si le gradient interne est negligeable (`Bi < 0,1`)."""
        return self.biot() < BIOT_MINCE

    def diffusivite(self) -> float:
        """`alpha = lambda / (rho c)`, en m^2/s."""
        return self.conductivite / self.capacite_volumique

    def fourier(self, duree: float) -> float:
        """`Fo = alpha t / L^2`, le temps rendu sans dimension."""
        return self.diffusivite() * duree / self.longueur_caracteristique**2

    def constante_de_temps(self) -> float:
        """`rho c L / K` ; leve si le corps n'est pas thermiquement mince."""
        if not self.est_thermiquement_mince():
            msg = (
                f"corps non thermiquement mince : Bi = {self.biot():.3f} >= "
                f"{BIOT_MINCE} -- la temperature uniforme est une fiction"
            )
            raise ValueError(msg)
        return self.capacite_volumique * self.longueur_caracteristique / self.echange

    def ecart_relatif(self, duree: float) -> float:
        """`exp(-t / tau)` : la decroissance du corps mince."""
        return math.exp(-duree / self.constante_de_temps())
