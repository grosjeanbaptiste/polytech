"""Le bruit gaussien : sa valeur efficace et son excursion.

Chapitre 4, partie I, planches 50 a 53. Trois proprietes servent partout :

* des bruits gaussiens independants s'ajoutent en **variance**, donc en
  carre des valeurs efficaces ;
* un bruit gaussien centre a pour ecart-type sa valeur efficace,
  `sigma_n = V_RMS` ;
* il n'est pas borne, mais tombe 99,9 % du temps dans `+-3,2905 sigma_n` :
  d'ou l'excursion crete a crete de `6,6 V_RMS` que les fiches techniques
  emploient.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.maths.domain.probabilites.loi_normale import (
    demi_largeur_de_l_intervalle,
)

PROBABILITE_DE_L_EXCURSION: Final = 0.999
"""La planche 53 : l'intervalle ou le bruit tombe « 99.9 % of the time »."""


def valeur_efficace_de_la_somme(*efficaces: float) -> float:
    """`racine(somme V_i^2)`, pour des bruits independants."""
    if any(efficace < 0 for efficace in efficaces):
        msg = f"valeur efficace negative : {efficaces}"
        raise ValueError(msg)
    return math.sqrt(sum(efficace**2 for efficace in efficaces))


def excursion_crete_a_crete(
    efficace: float, *, probabilite: float = PROBABILITE_DE_L_EXCURSION
) -> float:
    """`2 z V_RMS` ; `6,6 V_RMS` a 99,9 %."""
    if efficace < 0:
        msg = f"valeur efficace negative : {efficace}"
        raise ValueError(msg)
    return 2 * demi_largeur_de_l_intervalle(probabilite) * efficace
