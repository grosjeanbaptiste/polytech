"""Exercice cote 2019, question 3 : un noyau a trois colonnes et deux entrefers.

Le champ impose dans les entrefers fixe l'induction ; le flux se conserve
et se double dans la colonne centrale, deux fois plus large : l'induction
est la meme dans tout le fer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    potentiel_scalaire_magnetique,
)

CHAMP_D_ENTREFER: Final = 700e3
ENTREFER: Final = 2e-3
EPAISSEUR: Final = 2e-2
LARGEURS: Final = (2e-2, 4e-2, 4e-2, 4e-2, 2e-2)
"""Colonne gauche, fenetre, colonne centrale, fenetre, colonne droite."""
HAUTEURS: Final = (2e-2, 6e-2, 2e-2)
"""Culasse haute, fenetre, culasse basse."""
CHAMP_LU_DANS_LE_FER: Final = 160.0
"""`H`, en A/m, **lu sur la figure 3.2** (toles ordinaires) pour 0,88 T."""
INDUCTANCE_VOULUE: Final = 10.0


@dataclass(frozen=True, slots=True)
class Etat:
    """L'induction dans les entrefers, et les flux qui en decoulent."""

    induction: float
    flux_lateral: float
    flux_central: float
    induction_centrale: float


def etat() -> Etat:
    """`B = mu_0 H_e`, puis les flux par conservation."""
    induction = PERMEABILITE_DU_VIDE * CHAMP_D_ENTREFER
    lateral = induction * LARGEURS[0] * EPAISSEUR
    central = 2 * lateral
    return Etat(induction, lateral, central, central / (LARGEURS[2] * EPAISSEUR))


def longueurs() -> tuple[float, float]:
    """La colonne centrale, puis le fer du reste du tour d'une fenetre."""
    haute, fenetre, basse = HAUTEURS
    gauche, ouverture, centre = LARGEURS[:3]
    centrale = haute / 2 + fenetre + basse / 2
    culasse = centre / 2 + ouverture + gauche / 2
    return centrale, 2 * culasse + centrale - ENTREFER


def amperes_tours(champ: float = CHAMP_LU_DANS_LE_FER) -> tuple[float, float, float]:
    """Ce que consomment la colonne centrale, le reste du fer, puis l'entrefer."""
    centrale, laterale = longueurs()
    return (
        potentiel_scalaire_magnetique(champ, centrale),
        potentiel_scalaire_magnetique(champ, laterale),
        potentiel_scalaire_magnetique(CHAMP_D_ENTREFER, ENTREFER),
    )


def spires(
    flux_central: float | None = None, champ: float = CHAMP_LU_DANS_LE_FER
) -> float:
    """`N = racine(N I L / Phi_c)` : `N Phi_c = L I` et `N I` connu."""
    flux = etat().flux_central if flux_central is None else flux_central
    return float((sum(amperes_tours(champ)) * INDUCTANCE_VOULUE / flux) ** 0.5)
