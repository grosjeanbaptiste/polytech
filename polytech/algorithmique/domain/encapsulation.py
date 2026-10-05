"""L'encapsulation : qui a le droit de toucher a quel membre.

Le cours (seance 4, « Encapsulation », « Manipulation des membres prives »
et « Amies : fonctions et classes ») :

* un membre `private` « ne peut etre accede que par les autres membres de
  cette meme classe » ;
* un membre `protected` l'est en plus « par les membres des classes
  derivees » ;
* un membre `public` l'est « par toutes les autres classes et fonctions » ;
* sans mot cle, tout est public dans une `struct` et prive dans une
  `class` -- c'est leur seule difference ;
* une fonction ou une classe declaree `friend` recoit « les memes
  privileges que ceux des fonctions membres ». L'amitie se donne : elle
  n'est pas reciproque.

Un acces refuse est l'erreur C2248 des planches : le programme ne compile
pas. La notation UML des diagrammes de classes marque la visibilite d'un
signe : `-` prive, `#` protege, `+` public.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PRIVE: Final = "private"
PROTEGE: Final = "protected"
PUBLIC: Final = "public"
_SIGNES_UML: Final = {PRIVE: "-", PROTEGE: "#", PUBLIC: "+"}
_PAR_DEFAUT: Final = {"struct": PUBLIC, "class": PRIVE}


def visibilite_par_defaut(mot_cle: str) -> str:
    """La visibilite d'un membre declare sans mot cle."""
    if mot_cle not in _PAR_DEFAUT:
        msg = f"ni struct ni class : {mot_cle}"
        raise ValueError(msg)
    return _PAR_DEFAUT[mot_cle]


def ligne_uml(membre: str, type_du_membre: str, visibilite: str) -> str:
    """La ligne d'un diagramme de classes : signe, nom, type."""
    return f"{_SIGNES_UML[visibilite]}{membre} : {type_du_membre}"


@dataclass(frozen=True, slots=True)
class Classe:
    """Une classe : la visibilite de chaque membre, et ses amies."""

    nom: str
    membres: Mapping[str, str]
    amies: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Chaque membre porte l'une des trois visibilites."""
        for membre, visibilite in self.membres.items():
            if visibilite not in _SIGNES_UML:
                msg = f"visibilite inconnue pour {membre} : {visibilite}"
                raise ValueError(msg)

    @classmethod
    def de(cls, mot_cle: str, nom: str, membres: Sequence[str]) -> Classe:
        """Une `struct` ou une `class` dont aucun membre ne porte de mot cle."""
        return cls(nom, dict.fromkeys(membres, visibilite_par_defaut(mot_cle)))

    def accede(self, membre: str, depuis: str, *, derivee: bool = False) -> bool:
        """Vrai si le code de la fonction ou de la classe `depuis` peut y toucher."""
        if membre not in self.membres:
            msg = f"{membre} n'est pas membre de {self.nom}"
            raise AttributeError(msg)
        visibilite = self.membres[membre]
        if visibilite == PUBLIC or depuis == self.nom or depuis in self.amies:
            return True
        return visibilite == PROTEGE and derivee
