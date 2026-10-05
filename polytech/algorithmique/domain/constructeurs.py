"""Le choix d'un constructeur selon le nombre d'arguments.

Le cours (seance 4, « Constructeurs » et « Constructeur par defaut ») :
une classe peut proposer plusieurs constructeurs, et « la selection du bon
constructeur devient primordiale ». Trois regles :

* un constructeur accepte un nombre d'arguments compris entre celui de
  ses parametres sans valeur par defaut et celui de tous ses parametres ;
* un **constructeur par defaut** est celui qui accepte zero argument ;
  c'est lui qu'appellent `A Obj;` et chaque case de `A Tab[3];` ;
* sans aucun constructeur declare, le compilateur en fournit un, sans
  parametre. En declarer un seul le retire.

Deux constructeurs qui acceptent le meme appel le rendent ambigu. Le
choix selon le *type* des arguments est celui de toute surcharge, et vit
dans `surcharge` ; le constructeur de copie n'entre pas dans ce compte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Constructeur:
    """Un constructeur : ses parametres, et combien ont une valeur par defaut."""

    parametres: int
    par_defaut: int = 0

    def __post_init__(self) -> None:
        """Les valeurs par defaut ne portent que sur des parametres existants."""
        if not 0 <= self.par_defaut <= self.parametres:
            msg = f"valeurs par defaut hors de 0 a {self.parametres}"
            raise ValueError(msg)

    def accepte(self, arguments: int) -> bool:
        """Vrai si l'appel fournit assez d'arguments, et pas trop."""
        return self.parametres - self.par_defaut <= arguments <= self.parametres


IMPLICITE: Final = Constructeur(0)
"""Le constructeur que le compilateur fournit quand la classe n'en declare aucun."""


def constructeur_appele(
    constructeurs: Sequence[Constructeur], arguments: int
) -> Constructeur:
    """Le constructeur que choisit un appel ; leve s'il n'y en a aucun, ou deux."""
    declares = constructeurs or (IMPLICITE,)
    candidats = [c for c in declares if c.accepte(arguments)]
    if not candidats:
        msg = f"aucun constructeur n'accepte {arguments} argument(s)"
        raise TypeError(msg)
    if len(candidats) > 1:
        msg = f"appel ambigu : {len(candidats)} constructeurs l'acceptent"
        raise TypeError(msg)
    return candidats[0]


def est_constructible(constructeurs: Sequence[Constructeur], arguments: int) -> bool:
    """Vrai si un constructeur, et un seul, accepte l'appel."""
    declares = constructeurs or (IMPLICITE,)
    return sum(c.accepte(arguments) for c in declares) == 1


def tableau_constructible(constructeurs: Sequence[Constructeur]) -> bool:
    """`A Tab[n];` : chaque case demande un constructeur par defaut."""
    return est_constructible(constructeurs, 0)
