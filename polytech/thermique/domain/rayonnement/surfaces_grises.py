"""Les surfaces grises, et la resistance que leur emissivite oppose.

Le cours traite l'enceinte a surfaces grises par un schema en
resistances, exactement comme la conduction : une **resistance de
surface** `(1 - eps) / (eps S)` par surface, et une **resistance
geometrique** `1 / (S_i F_ij)` par couple. L'analogie est ce qui rend le
chapitre calculable, et c'est elle qui est portee ici.

Une surface noire (`eps = 1`) a une resistance de surface nulle : elle ne
freine rien. Une surface d'emissivite nulle aurait une resistance infinie,
donc n'echangerait rien -- et le dire en levant vaut mieux que de rendre
un infini que le document composerait.
"""

from __future__ import annotations

from dataclasses import dataclass

from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.validation import exiger_strictement_positif


@dataclass(frozen=True, slots=True)
class SurfaceGrise:
    """Une surface grise : son aire, son emissivite, sa temperature."""

    aire: float
    emissivite: float
    temperature: float

    def __post_init__(self) -> None:
        """Refuse une aire nulle et une emissivite hors de ]0, 1]."""
        exiger_strictement_positif(self.aire, "aire de surface grise")
        if not 0 < self.emissivite <= 1:
            msg = (
                f"emissivite hors de ]0, 1] : {self.emissivite} -- une "
                f"emissivite nulle opposerait une resistance infinie"
            )
            raise ValueError(msg)

    def resistance_de_surface(self) -> float:
        """`(1 - eps) / (eps S)` : nulle pour une surface noire."""
        return (1 - self.emissivite) / (self.emissivite * self.aire)

    def emittance_de_corps_noir(self) -> float:
        """`sigma T^4`, le potentiel du schema en resistances."""
        return emittance(self.temperature)


def resistance_geometrique(aire: float, facteur: float) -> float:
    """`1 / (S_i F_ij)` ; leve si le facteur d'angle est nul."""
    exiger_strictement_positif(aire, "aire")
    exiger_strictement_positif(facteur, "facteur d'angle")
    return 1.0 / (aire * facteur)


def flux_entre_deux_surfaces(
    premiere: SurfaceGrise,
    seconde: SurfaceGrise,
    facteur: float,
) -> float:
    """Le flux net echange, par le schema en resistances en serie."""
    resistance = (
        premiere.resistance_de_surface()
        + resistance_geometrique(premiere.aire, facteur)
        + seconde.resistance_de_surface()
    )
    ecart = premiere.emittance_de_corps_noir() - seconde.emittance_de_corps_noir()
    return ecart / resistance


def flux_entre_deux_plans_gris(
    premiere_emissivite: float,
    seconde_emissivite: float,
    premiere_temperature: float,
    seconde_temperature: float,
    aire: float,
) -> float:
    """`sigma S (T1^4 - T2^4) / (1/eps_1 + 1/eps_2 - 1)`.

    Cas particulier que le fascicule tabule, et qui sert partout : deux
    vitres, deux parois d'un four, un ecran anti-rayonnement. Le `-1` du
    denominateur est ce qui distingue deux plans infinis d'une enceinte
    quelconque -- chacun ne voit que l'autre, donc le facteur d'angle vaut
    un dans les deux sens.
    """
    for emissivite in (premiere_emissivite, seconde_emissivite):
        if not 0 < emissivite <= 1:
            msg = f"emissivite hors de ]0, 1] : {emissivite}"
            raise ValueError(msg)
    if aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2"
        raise ValueError(msg)
    denominateur = 1 / premiere_emissivite + 1 / seconde_emissivite - 1
    ecart = emittance(premiere_temperature) - emittance(seconde_temperature)
    return aire * ecart / denominateur


def resistance_d_une_surface_adiabatique() -> float:
    """Zero : une surface adiabatique reradie tout ce qu'elle recoit.

    Le fascicule lui consacre une planche -- « 3 surfaces dont 1 surface
    adiabatique » -- parce qu'elle modifie le schema en resistances sans
    ajouter d'inconnue de temperature : son flux net est nul, donc elle ne
    porte aucune resistance de surface.
    """
    return 0.0
