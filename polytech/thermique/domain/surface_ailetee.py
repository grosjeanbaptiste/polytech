"""Le coefficient global d'une paroi garnie d'ailettes.

Syllabus de thermique, tome 2, deuxieme partie, section 1.6. Un gaz
echange mal ; on compense en multipliant la surface de son cote. Pour
une surface primaire `S_p` -- la paroi avant les ailettes -- le cote
ailete offre la fraction restee libre entre les pieds, `S_1o`, et les
ailettes, `S_1a`, qui ne valent que leur efficacite `Omega` :

    1/K_gl,p = 1/K_2 + e/lambda_p + S_p / ((S_1o + Omega S_1a) K_1).

Le terme `e/lambda_p` suppose le flux unidimensionnel dans la paroi, ce
que les concentrations au pied des ailettes dementent ; il est le plus
souvent negligeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SurfaceAiletee:
    """`S_p`, `S_1o` et `S_1a`, en m^2, pour une meme portion de paroi."""

    primaire: float
    libre: float
    ailettes: float

    def __post_init__(self) -> None:
        """Refuse une surface primaire nulle et toute surface negative."""
        if self.primaire <= 0:
            msg = f"surface primaire nulle ou negative : {self.primaire} m^2"
            raise ValueError(msg)
        if self.libre < 0 or self.ailettes < 0:
            msg = f"surface negative : {self.libre} ou {self.ailettes} m^2"
            raise ValueError(msg)

    @property
    def totale(self) -> float:
        """`S_1t = S_1a + S_1o`."""
        return self.libre + self.ailettes

    def resistance(self, echange: float, efficacite: float) -> float:
        """`S_p / ((S_1o + Omega S_1a) K_1)`, rapportee a la surface primaire."""
        if not 0 < efficacite <= 1:
            msg = f"efficacite hors de ]0, 1] : {efficacite}"
            raise ValueError(msg)
        utile = self.libre + efficacite * self.ailettes
        return self.primaire / (utile * echange)


def coefficient_global_de_surface_ailetee(
    cote_lisse: float,
    cote_ailete: float,
    *,
    paroi: tuple[float, float],
    surfaces: SurfaceAiletee,
    efficacite: float,
) -> float:
    """`K_gl,p` d'une paroi plane ; `paroi = (e, lambda_p)`."""
    epaisseur, conductivite = paroi
    ailete = surfaces.resistance(cote_ailete, efficacite)
    return 1 / (1 / cote_lisse + epaisseur / conductivite + ailete)


def coefficient_global_de_tube_ailete(  # noqa: PLR0913 -- les six grandeurs de la formule
    interieur: float,
    exterieur: float,
    *,
    rayons: tuple[float, float],
    conductivite: float,
    surfaces: SurfaceAiletee,
    efficacite: float,
) -> float:
    """`K_gl,i` d'un tube a ailettes exterieures ; `surfaces.primaire` vaut `S_i`."""
    interne, externe = rayons
    paroi = interne / conductivite * math.log(externe / interne)
    return 1 / (1 / interieur + paroi + surfaces.resistance(exterieur, efficacite))
