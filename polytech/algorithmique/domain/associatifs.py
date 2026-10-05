"""Le tableau associatif : `set`, `multiset`, `map`, `multimap`.

Le cours (note « Les fonctions membres », pages 4 et 5 ; seance 5-6,
planches 105 et 106) : quatre conteneurs associatifs, des « ensembles
ordonnes » ranges par leur cle, en ordre croissant par defaut.

* `set <K>` ne garde qu'un exemplaire de chaque cle ;
* `multiset` est « un set acceptant des duplications d'elements » ;
* `map <K, V>` associe une valeur a chaque cle, unique ;
* `multimap` est « un map acceptant des duplications de cles ».

Quel que soit l'ordre des insertions, le parcours se fait par cle
croissante. Seul `map` offre l'acces direct `at(cle)`.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any

from polytech.algorithmique.domain.conteneurs import ASSOCIATIFS, exiger


class Associatif:
    """Un conteneur associatif : des cles triees, et leurs valeurs."""

    def __init__(self, genre: str) -> None:
        """Un conteneur vide de l'un des quatre genres."""
        if genre not in ASSOCIATIFS:
            msg = f"conteneur associatif inconnu : {genre}"
            raise ValueError(msg)
        self.genre = genre
        self._cles: list[Any] = []
        self._valeurs: list[Any] = []

    def insert(self, cle: Any, valeur: Any = None) -> bool:  # noqa: ANN401 -- tout type ordonne
        """Range la cle a sa place ; faux si elle est deja la et doit etre unique."""
        if not self.genre.startswith("multi") and self.count(cle):
            return False
        rang = bisect_right(self._cles, cle)
        self._cles.insert(rang, cle)
        self._valeurs.insert(rang, valeur)
        return True

    def cles(self) -> tuple[Any, ...]:
        """Les cles dans l'ordre du parcours : croissant."""
        return tuple(self._cles)

    def size(self) -> int:
        """Le nombre d'elements."""
        return len(self._cles)

    def count(self, cle: Any) -> int:  # noqa: ANN401 -- le type des cles
        """Le nombre d'elements de cette cle."""
        return self.upper_bound(cle) - self.lower_bound(cle)

    def lower_bound(self, cle: Any) -> int:  # noqa: ANN401 -- le type des cles
        """La position de la premiere cle superieure ou egale."""
        return bisect_left(self._cles, cle)

    def upper_bound(self, cle: Any) -> int:  # noqa: ANN401 -- le type des cles
        """La position de la premiere cle strictement superieure."""
        return bisect_right(self._cles, cle)

    def at(self, cle: Any) -> Any:  # noqa: ANN401 -- le type des valeurs
        """La valeur associee a la cle -- du `map` seulement."""
        exiger(self.genre, "at")
        if not self.count(cle):
            msg = f"cle absente du map : {cle}"
            raise KeyError(msg)
        return self._valeurs[self.lower_bound(cle)]

    def affecter(self, cle: Any, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """`conteneur[cle] = valeur` : ajoute, ou ecrase -- du `map` seulement."""
        exiger(self.genre, "operator[]")
        if not self.insert(cle, valeur):
            self._valeurs[self.lower_bound(cle)] = valeur

    def valeurs_de(self, cle: Any) -> tuple[Any, ...]:  # noqa: ANN401 -- le type des cles
        """Toutes les valeurs de cette cle, dans l'ordre de leur insertion."""
        return tuple(self._valeurs[self.lower_bound(cle) : self.upper_bound(cle)])

    def erase(self, cle: Any) -> int:  # noqa: ANN401 -- le type des cles
        """Ote tous les elements de cette cle, et rend leur nombre."""
        debut, fin = self.lower_bound(cle), self.upper_bound(cle)
        del self._cles[debut:fin]
        del self._valeurs[debut:fin]
        return fin - debut

    def clear(self) -> None:
        """Ote tous les elements."""
        self._cles.clear()
        self._valeurs.clear()
