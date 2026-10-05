"""L'heritage public et le polymorphisme.

Le cours (seance 5, « L'heritage » et « Notion de polymorphisme ») : une
classe derivee herite des attributs et des methodes de sa classe de base ;
elle « est une » specialisation de celle-ci. Cinq regles s'en calculent :

* un membre se cherche d'abord dans la classe, puis dans ses bases : une
  redefinition **masque** la fonction heritee. Atteint par deux chemins
  -- heritage multiple --, il est ambigu ;
* le **prive** d'une base reste ferme a sa derivee ; le **protege** lui
  est ouvert, et ferme au code exterieur ;
* a la construction, les bases passent d'abord, dans l'ordre de leur
  declaration, puis les objets membres, puis la classe ; la destruction
  suit l'ordre inverse ;
* sans `virtual`, la fonction appelee a travers un pointeur de base est
  celle du **type du pointeur** ;
* avec `virtual`, c'est celle du **type de l'objet** designe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polytech.algorithmique.domain.encapsulation import PRIVE, PUBLIC

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class Declaration:
    """Ce qu'une classe declare : ses bases, ses membres, ses objets membres."""

    meres: tuple[str, ...] = ()
    membres: Mapping[str, str] = field(default_factory=dict)
    virtuelles: frozenset[str] = frozenset()
    objets_membres: tuple[str, ...] = ()


class Hierarchie:
    """Des classes liees par un heritage public."""

    def __init__(self, declarations: Mapping[str, Declaration]) -> None:
        """Chaque base et chaque type d'objet membre doit etre declare."""
        for nom, declaration in declarations.items():
            for autre in (*declaration.meres, *declaration.objets_membres):
                if autre not in declarations:
                    msg = f"classe de base inconnue pour {nom} : {autre}"
                    raise ValueError(msg)
        self._declarations = dict(declarations)
        self.classes = tuple(declarations)

    def _declaration(self, classe: str) -> Declaration:
        if classe not in self._declarations:
            msg = f"classe inconnue : {classe}"
            raise KeyError(msg)
        return self._declarations[classe]

    def _chemins(self, classe: str, membre: str) -> list[str]:
        declaration = self._declaration(classe)
        if membre in declaration.membres:
            return [classe]
        return [c for mere in declaration.meres for c in self._chemins(mere, membre)]

    def _ancetres(self, classe: str) -> set[str]:
        meres = self._declaration(classe).meres
        return set(meres).union(*(self._ancetres(mere) for mere in meres))

    def definit(self, classe: str, membre: str) -> str:
        """La classe dont le membre est employe par un objet de cette classe."""
        chemins = self._chemins(classe, membre)
        if not chemins:
            msg = f"la classe {classe} n'a pas de membre {membre}"
            raise AttributeError(msg)
        if len(chemins) > 1:
            msg = f"membre ambigu : {membre} est herite par {len(chemins)} chemins"
            raise TypeError(msg)
        return chemins[0]

    def accede(self, classe: str, membre: str, depuis: str) -> bool:
        """Vrai si le code de `depuis` peut toucher ce membre d'un objet."""
        origine = self.definit(classe, membre)
        visibilite = self._declarations[origine].membres[membre]
        if visibilite == PUBLIC or depuis == origine:
            return True
        derivee = depuis in self._declarations and origine in self._ancetres(depuis)
        return visibilite != PRIVE and derivee

    def ordre_de_construction(self, classe: str) -> tuple[str, ...]:
        """Les constructeurs appeles, du premier au dernier."""
        declaration = self._declaration(classe)
        avant = (*declaration.meres, *declaration.objets_membres)
        return (*(c for a in avant for c in self.ordre_de_construction(a)), classe)

    def ordre_de_destruction(self, classe: str) -> tuple[str, ...]:
        """Les destructeurs appeles : l'ordre inverse de la construction."""
        return self.ordre_de_construction(classe)[::-1]

    def methode_executee(self, statique: str, dynamique: str, methode: str) -> str:
        """La classe dont la fonction s'execute par un pointeur de type `statique`."""
        if dynamique != statique and statique not in self._ancetres(dynamique):
            msg = f"{dynamique} n'est pas une classe derivee de {statique}"
            raise TypeError(msg)
        lignee = {statique} | self._ancetres(statique)
        virtuelle = any(methode in self._declarations[c].virtuelles for c in lignee)
        return self.definit(dynamique if virtuelle else statique, methode)
