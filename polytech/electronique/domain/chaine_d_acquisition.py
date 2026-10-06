"""Le budget de bruit d'une chaine d'acquisition, etage par etage.

Chaque etage -- amplificateur, filtre, multiplexeur, convertisseur -- a
un gain et un bruit propre, que sa fiche donne *rapporte a son entree*,
en volts par racine de hertz. Ce bruit traverse donc l'etage lui-meme,
puis tous ceux qui le suivent.

Deux lectures du meme budget :

* **en sortie** : `densite^2 x bande x (gain restant a traverser)^2` ;
* **a l'entree** : la meme puissance divisee par le carre du gain total,
  soit `densite^2 x bande / (gain deja traverse)^2`. C'est celle qui se
  compare au signal du capteur, et elle dit pourquoi le premier etage
  compte plus que tous les autres.

Les bruits des etages sont independants : leurs puissances s'ajoutent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Etage:
    """Un bloc de la chaine : son gain en tension, son bruit a son entree."""

    nom: str
    gain: float
    densite: float
    """La densite efficace du bruit en tension, en volts par racine de hertz."""

    def __post_init__(self) -> None:
        """Refuse un gain nul et une densite negative."""
        if self.gain <= 0:
            msg = f"gain nul ou negatif pour {self.nom} : {self.gain}"
            raise ValueError(msg)
        if self.densite < 0:
            msg = f"densite negative pour {self.nom} : {self.densite}"
            raise ValueError(msg)


def _exiger(chaine: Sequence[Etage], bande: float) -> None:
    if not chaine:
        msg = "chaine vide : aucun etage"
        raise ValueError(msg)
    if bande <= 0:
        msg = f"bande nulle ou negative : {bande} Hz"
        raise ValueError(msg)


def gain_total(chaine: Sequence[Etage]) -> float:
    """Le produit des gains, du capteur au convertisseur."""
    _exiger(chaine, 1.0)
    return math.prod(etage.gain for etage in chaine)


def puissances_en_sortie(chaine: Sequence[Etage], bande: float) -> list[float]:
    """La puissance de bruit de chaque etage en bout de chaine, en volts carres."""
    _exiger(chaine, bande)
    puissances = []
    for rang, etage in enumerate(chaine):
        restant = math.prod(suivant.gain for suivant in chaine[rang:])
        puissances.append(etage.densite**2 * bande * restant**2)
    return puissances


def puissances_a_l_entree(chaine: Sequence[Etage], bande: float) -> list[float]:
    """Les memes, rapportees a l'entree de la chaine."""
    total = gain_total(chaine) ** 2
    return [puissance / total for puissance in puissances_en_sortie(chaine, bande)]


def bruit_efficace(puissances: Sequence[float]) -> float:
    """La tension efficace de la somme de bruits independants."""
    if not puissances:
        msg = "aucune puissance a sommer"
        raise ValueError(msg)
    if any(puissance < 0 for puissance in puissances):
        msg = f"puissance negative : {list(puissances)}"
        raise ValueError(msg)
    return math.sqrt(sum(puissances))
