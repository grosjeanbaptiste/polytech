"""La duree de vie des objets : qui est construit, qui est detruit, et quand.

Le cours (seance 4, « Constructeur et destructeur » et « Regles de
visibilite des objets ») : le constructeur est appele implicitement a toute
creation d'instance, le destructeur a toute disparition. L'ordre est fixe
par l'endroit ou l'objet est declare :

* un objet **global** est construit avant le debut de `main` ;
* un objet **local** est construit a sa declaration et detruit a la
  fermeture de son bloc, le dernier construit le premier ;
* un objet local **statique** n'est construit qu'au premier passage ;
* a la fin du programme, globaux et statiques sont detruits dans l'ordre
  inverse de leur construction, apres les locaux de `main` ;
* un parametre passe **par valeur** est construit par copie, et detruit a
  la sortie de la fonction ; une reference ne construit rien ;
* un objet cree par `new` n'est detruit que par `delete`.
"""

from __future__ import annotations

from typing import Final

CONSTRUCTION: Final = "construction"
COPIE: Final = "copie"
DESTRUCTION: Final = "destruction"


class Execution:
    """Le journal des constructions et des destructions d'un programme."""

    def __init__(self) -> None:
        """Avant `main` : aucun bloc, aucun objet."""
        self.journal: list[tuple[str, str]] = []
        self._permanents: list[str] = []
        self._blocs: list[list[str]] = []
        self._tas: list[str] = []

    def _bloc(self) -> list[str]:
        if not self._blocs:
            msg = "aucun bloc ouvert : hors de toute fonction"
            raise RuntimeError(msg)
        return self._blocs[-1]

    def globale(self, nom: str) -> None:
        """Un objet declare hors de toute fonction."""
        if self._blocs:
            msg = "une variable globale se construit avant le debut de main"
            raise RuntimeError(msg)
        self.journal.append((CONSTRUCTION, nom))
        self._permanents.append(nom)

    def ouvrir(self) -> None:
        """Une accolade ouvrante : `main`, une fonction, un tour de boucle."""
        self._blocs.append([])

    def locale(self, nom: str, modele: str | None = None) -> None:
        """Un objet du bloc courant, copie d'un modele s'il en a un."""
        bloc = self._bloc()
        connus = {n for _, n in self.journal}
        if modele is not None and modele not in connus:
            msg = f"modele inconnu : {modele}"
            raise NameError(msg)
        self.journal.append((CONSTRUCTION if modele is None else COPIE, nom))
        bloc.append(nom)

    def tableau(self, nom: str, taille: int) -> None:
        """`T nom[taille];` : une construction par case, dans l'ordre."""
        for rang in range(taille):
            self.locale(f"{nom}[{rang}]")

    def statique(self, nom: str) -> None:
        """`static T nom;` : construit au premier passage seulement."""
        self._bloc()
        if nom not in self._permanents:
            self.journal.append((CONSTRUCTION, nom))
            self._permanents.append(nom)

    def fermer(self) -> None:
        """Une accolade fermante : les objets du bloc, en ordre inverse."""
        self._bloc()
        for nom in reversed(self._blocs.pop()):
            self.journal.append((DESTRUCTION, nom))

    def nouveau(self, nom: str) -> None:
        """`new T` : l'objet vit sur le tas, hors de tout bloc."""
        self.journal.append((CONSTRUCTION, nom))
        self._tas.append(nom)

    def supprimer(self, nom: str) -> None:
        """`delete` : la seule destruction d'un objet du tas."""
        if nom not in self._tas:
            msg = f"delete d'un objet absent du tas : {nom}"
            raise RuntimeError(msg)
        self._tas.remove(nom)
        self.journal.append((DESTRUCTION, nom))

    def terminer(self) -> None:
        """La fin du programme : les blocs restants, puis globaux et statiques."""
        while self._blocs:
            self.fermer()
        while self._permanents:
            self.journal.append((DESTRUCTION, self._permanents.pop()))

    @property
    def vivants(self) -> int:
        """Le nombre d'objets construits et pas encore detruits."""
        detruits = sum(evenement == DESTRUCTION for evenement, _ in self.journal)
        return len(self.journal) - 2 * detruits

    @property
    def fuites(self) -> tuple[str, ...]:
        """Les objets du tas qu'aucun `delete` n'a detruits."""
        return tuple(self._tas)
