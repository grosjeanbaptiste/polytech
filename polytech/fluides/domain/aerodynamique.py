"""L'aerodynamique : portance, trainee, et leurs coefficients.

Les planches definissent le coefficient de trainee par
`C_D = F_D / (rho V^2 L D / 2)`, c'est-a-dire la force rapportee a la
pression dynamique et a une surface de reference. La portance se definit
de meme. Ce sont deux **definitions**, et c'est pourquoi elles sont
inversibles : connaissant le coefficient on remonte a la force, et
reciproquement -- un test verifie l'aller-retour.

Le piege du chapitre est la **surface de reference** : elle n'est pas la
meme pour la portance (surface de l'aile vue de dessus) et pour la
trainee d'un corps (sa section frontale). Comparer deux coefficients
rapportes a deux surfaces differentes n'a aucun sens, et c'est pourquoi
la surface est un champ de l'objet et non un argument de passage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from polytech.fluides.domain.grandeurs import MasseVolumique


@dataclass(frozen=True, slots=True)
class Ecoulement:
    """Les conditions amont : masse volumique et vitesse."""

    fluide: MasseVolumique
    vitesse: float

    def __post_init__(self) -> None:
        """Refuse une vitesse nulle : la pression dynamique serait nulle."""
        if self.vitesse <= 0:
            msg = f"vitesse amont nulle ou negative : {self.vitesse} m/s"
            raise ValueError(msg)

    def pression_dynamique(self) -> float:
        """`rho V^2 / 2`, le denominateur de tous les coefficients."""
        return self.fluide.kg_par_m3 * self.vitesse**2 / 2


@dataclass(frozen=True, slots=True)
class SurfacePortante:
    """Un corps place dans l'ecoulement, avec sa surface de reference.

    La surface est portee par l'objet, et non passee a chaque appel :
    comparer deux coefficients rapportes a deux surfaces differentes n'a
    aucun sens, et ce choix rend la faute impossible.
    """

    ecoulement: Ecoulement
    surface_de_reference: float

    def __post_init__(self) -> None:
        """Refuse une surface de reference nulle."""
        if self.surface_de_reference <= 0:
            msg = (
                f"surface de reference nulle ou negative : {self.surface_de_reference}"
            )
            raise ValueError(msg)

    def _echelle(self) -> float:
        return self.ecoulement.pression_dynamique() * self.surface_de_reference

    def coefficient(self, force: float) -> float:
        """Le coefficient adimensionnel correspondant a cette force."""
        return force / self._echelle()

    def force(self, coefficient: float) -> float:
        """La force correspondant a ce coefficient : l'inverse exact."""
        return coefficient * self._echelle()

    def finesse(
        self, coefficient_de_portance: float, coefficient_de_trainee: float
    ) -> float:
        """`C_L / C_D`, la finesse -- ce que l'on cherche a maximiser."""
        if coefficient_de_trainee <= 0:
            msg = f"coefficient de trainee nul ou negatif : {coefficient_de_trainee}"
            raise ValueError(msg)
        return coefficient_de_portance / coefficient_de_trainee


RENDEMENT_D_ENVERGURE: Final = 0.85
"""`e` dans `C_Di = C_L^2 / (pi e AR)`. Valeur usuelle d'une aile reelle."""


def trainee_induite(
    portance: float,
    allongement: float,
    rendement: float = RENDEMENT_D_ENVERGURE,
) -> float:
    """`C_Di = C_L^2 / (pi e AR)` : la trainee qu'un profil fini ne peut eviter.

    Elle vient des tourbillons de bout d'aile, donc elle n'existe pas sur
    un profil infini. Elle croit comme le **carre** de la portance et
    decroit comme l'allongement -- ce qui explique d'un seul coup les
    grandes ailes fines des planeurs et la traction des avions lourds a
    basse vitesse.
    """
    if allongement <= 0:
        msg = f"allongement nul ou negatif : {allongement}"
        raise ValueError(msg)
    if not 0 < rendement <= 1:
        msg = f"rendement d'envergure hors de ]0, 1] : {rendement}"
        raise ValueError(msg)
    return portance**2 / (math.pi * rendement * allongement)
