"""Les milieux aimantes : permeabilite, susceptibilite, temperature de Curie.

Le syllabus d'electricite generale (10.1 a 10.3). Tous les corps sont
diamagnetiques : l'aimantation `M_d` qu'un champ y fait naitre lui est
**opposee**. Ceux dont l'atome porte un moment magnetique sont en outre
paramagnetiques, d'une aimantation `M_p` de meme sens que le champ, qui
l'emporte. Sous leur temperature de Curie, certains de ces derniers sont
ferromagnetiques.

`B = mu_0 (H + M)` (10.3.2) s'ecrit `B = mu H` (10.3.3) ; la permeabilite
relative `mu_r = mu / mu_0` (10.3.4) et la susceptibilite `chi`, definie
par `M = chi H` (10.3.5), sont liees par `chi = mu_r - 1` (10.3.6).
"""

from __future__ import annotations

from typing import Final

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE

CORPS_DIAMAGNETIQUES: Final = {
    "cuivre": 0.999990,
    "eau": 0.999991,
    "bismuth": 0.99984,
}
"""Tableau 10.3.1, premiere moitie : permeabilites relatives sous l'unite."""

CORPS_PARAMAGNETIQUES: Final = {
    "aluminium": 1.000020,
    "air": 1.0000004,
    "oxygene": 1.000002,
    "FeCl3": 1.003,
}
"""Tableau 10.3.1, seconde moitie : permeabilites relatives au-dessus."""

TEMPERATURE_DE_CURIE_DU_FER: Final = 1050.0
"""En kelvins, « de l'ordre de 1 050 K » (page 10-4)."""


def susceptibilite(relative: float) -> float:
    """`chi = mu_r - 1` (10.3.6), sans dimension."""
    if relative <= 0:
        msg = f"permeabilite relative nulle ou negative : {relative}"
        raise ValueError(msg)
    return relative - 1


def aimantation_induite(relative: float, champ: float) -> float:
    """`M = chi H` (10.3.5), en amperes par metre : negative si diamagnetique."""
    return susceptibilite(relative) * champ


def induction_du_milieu(champ: float, aimantation: float) -> float:
    """`B = mu_0 (H + M)` (10.3.2), en teslas."""
    return PERMEABILITE_DU_VIDE * (champ + aimantation)


def est_ferromagnetique(temperature: float, curie: float) -> bool:
    """Vrai sous la temperature de Curie, ou l'agitation thermique cede (10.2)."""
    if temperature < 0:
        msg = f"temperature absolue negative : {temperature} K"
        raise ValueError(msg)
    return temperature < curie
