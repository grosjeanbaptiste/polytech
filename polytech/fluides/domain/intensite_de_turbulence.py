"""L'intensite de turbulence d'un ecoulement.

Les trois composantes de la vitesse fluctuent autour de leur moyenne ;
l'intensite rapporte l'ecart quadratique moyen de ces fluctuations a une
vitesse de reference :

    J = racine((v_x'^2 + v_y'^2 + v_z'^2) / 3) / u_0.

Les planches de convection forcee de la thermique (jeu 09, planche 22)
la citent parmi les variables « mal connues » dont depend la similitude :
deux essais au meme Reynolds ne se ressemblent que si elle est la meme.
Elle vaut `racine(2 k / 3) / u_0`, ou `k` est l'energie cinetique
turbulente.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from polytech.fluides.domain.paroi_turbulente import energie_cinetique_turbulente

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.fluides.domain.turbulence import Decomposition


def intensite_de_turbulence(
    composantes: Sequence[Decomposition], reference: float
) -> float:
    """`J`, sans dimension, pour les trois composantes echantillonnees."""
    if reference <= 0:
        msg = f"vitesse de reference nulle ou negative : {reference} m/s"
        raise ValueError(msg)
    energie = energie_cinetique_turbulente(composantes)
    return math.sqrt(2 * energie / 3) / reference
