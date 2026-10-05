"""Les deux conteneurs sequentiels du cours : `vector` et `list`.

Le cours (seances 5 et 6) : `vector` generalise le tableau -- acces direct
par indice, ajout et retrait par la fin ; `list` est une liste doublement
chainee -- ajout et retrait aux deux bouts, pas d'indice. Chacun n'offre
que les fonctions membres de sa colonne du tableau (`conteneurs`).

Une position joue ici le role de l'iterateur : 0 pour `begin()`, la taille
pour `end()`. Deux regles s'y lisent, que le cours enseigne par l'exemple :

* `insert(position, valeur)` place l'element **devant** la position ;
* `erase(position)` rend la position de l'element **suivant** -- la meme
  valeur, puisque tout ce qui suit a recule d'un cran. C'est ce qui rend
  fautive la boucle `for (...; it++) { it = V.erase(it); }`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polytech.algorithmique.domain.algorithmes_stl import remove, unique
from polytech.algorithmique.domain.conteneurs import exiger
from polytech.algorithmique.domain.recherche_triee import merge

if TYPE_CHECKING:
    from collections.abc import Iterable


class Sequence:
    """Un `vector` ou une `list`, avec les seules fonctions membres qu'il a."""

    def __init__(self, genre: str, elements: Iterable[Any] = ()) -> None:
        """Le conteneur et son contenu initial."""
        if genre not in ("vector", "list"):
            msg = f"ni vector ni list : {genre}"
            raise ValueError(msg)
        self.genre = genre
        self._e = list(elements)

    @classmethod
    def de_taille(cls, genre: str, taille: int, valeur: Any = 0) -> Sequence:  # noqa: ANN401 -- tout type
        """`vector<T> v(taille, valeur)` : autant d'exemplaires de la valeur."""
        return cls(genre, [valeur] * taille)

    @property
    def elements(self) -> tuple[Any, ...]:
        """Le contenu, de `begin()` a `end()`."""
        return tuple(self._e)

    def _non_vide(self, membre: str) -> None:
        exiger(self.genre, membre)
        if not self._e:
            msg = f"conteneur vide : {membre} n'y a pas de sens"
            raise IndexError(msg)

    def size(self) -> int:
        """Le nombre d'elements."""
        return len(self._e)

    def empty(self) -> bool:
        """Vrai si le conteneur n'a aucun element."""
        return not self._e

    def front(self) -> Any:  # noqa: ANN401 -- le type des elements
        """Le premier element."""
        self._non_vide("front")
        return self._e[0]

    def back(self) -> Any:  # noqa: ANN401 -- le type des elements
        """Le dernier element."""
        self._non_vide("back")
        return self._e[-1]

    def at(self, indice: int) -> Any:  # noqa: ANN401 -- le type des elements
        """L'acces par indice verifie -- du vecteur seulement."""
        exiger(self.genre, "at")
        if not 0 <= indice < len(self._e):
            msg = f"indice hors du vecteur : {indice}"
            raise IndexError(msg)
        return self._e[indice]

    def push_back(self, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """Ajoute a la fin."""
        self._e.append(valeur)

    def push_front(self, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """Ajoute au debut -- de la liste seulement."""
        exiger(self.genre, "push_front")
        self._e.insert(0, valeur)

    def pop_back(self) -> None:
        """Retire le dernier element."""
        self._non_vide("pop_back")
        self._e.pop()

    def pop_front(self) -> None:
        """Retire le premier element -- de la liste seulement."""
        self._non_vide("pop_front")
        self._e.pop(0)

    def insert(self, position: int, valeur: Any, nombre: int = 1) -> None:  # noqa: ANN401 -- tout type
        """Insere `nombre` exemplaires devant la position."""
        if not 0 <= position <= len(self._e):
            msg = f"position hors du conteneur : {position}"
            raise IndexError(msg)
        self._e[position:position] = [valeur] * nombre

    def erase(self, debut: int, fin: int | None = None) -> int:
        """Efface de `debut` a `fin` exclu ; rend la position de l'element suivant."""
        fin = debut + 1 if fin is None else fin
        if not 0 <= debut <= fin <= len(self._e):
            msg = f"plage hors du conteneur : {debut} a {fin}"
            raise IndexError(msg)
        del self._e[debut:fin]
        return debut

    def remove(self, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """Retire toutes les occurrences -- de la liste seulement."""
        exiger(self.genre, "remove")
        del self._e[remove(self._e, valeur) :]

    def unique(self) -> None:
        """Ne garde que le premier de chaque groupe d'elements consecutifs egaux."""
        exiger(self.genre, "unique")
        del self._e[unique(self._e) :]

    def sort(self, *, decroissant: bool = False) -> None:
        """Range en ordre croissant, ou decroissant -- la liste seulement."""
        exiger(self.genre, "sort")
        self._e.sort(reverse=decroissant)

    def reverse(self) -> None:
        """Inverse l'ordre des elements -- la liste seulement."""
        exiger(self.genre, "reverse")
        self._e.reverse()

    def merge(self, autre: Sequence) -> None:
        """Fusionne une autre liste triee dans celle-ci, et la vide."""
        exiger(self.genre, "merge")
        self._e = merge(self._e, autre._e)
        autre._e.clear()

    def splice(self, position: int, autre: Sequence) -> None:
        """Deplace tous les elements de l'autre liste devant la position."""
        exiger(self.genre, "splice")
        for rang, element in enumerate(autre.elements):
            self.insert(position + rang, element)
        autre._e.clear()

    def resize(self, taille: int, valeur: Any = 0) -> None:  # noqa: ANN401 -- tout type
        """Tronque, ou complete avec la valeur."""
        self._e = [*self._e, *[valeur] * (taille - len(self._e))][:taille]

    def clear(self) -> None:
        """Efface tous les elements."""
        self._e.clear()
