"""Le tableau statique : une capacite fixee, et un nombre d'elements qui varie.

Le cours (partie 1, planches 55 a 62) : `int tab[Nmax]` reserve `Nmax`
cases, numerotees de 0 a `Nmax - 1`, dont les `N` premieres servent. Le
langage ne verifie pas les bornes : ecrire en `tab[Nmax]` ou en `tab[-1]`
compile, et ecrase la memoire voisine.

* **supprimer** l'element de rang `p`, c'est recopier chaque suivant sur
  son precedent, puis diminuer `N` ;
* **ajouter** au rang `p`, c'est d'abord decaler vers la droite, en
  partant de la fin, puis ecrire, puis augmenter `N` -- s'il reste de la
  place.

Ce module refuse ce que le C++ laisse passer : une position hors du
tableau, un ajout dans un tableau plein.

La boucle sur les elements -- `for (auto elem : tab)` -- travaille sur une
**copie** de chaque case ; avec `auto&`, sur la case elle-meme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def indice_valide(indice: int, dimension: int) -> bool:
    """Vrai pour un indice de 0 a `dimension - 1`."""
    return 0 <= indice < dimension


@dataclass(slots=True)
class TableauStatique:
    """`capacite` cases reservees, dont `taille` sont occupees."""

    capacite: int
    elements: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Refuse plus d'elements que de cases."""
        if len(self.elements) > self.capacite:
            msg = "plus d'elements que de places dans le tableau"
            raise OverflowError(msg)

    @property
    def taille(self) -> int:
        """`N`, le nombre de cases occupees."""
        return len(self.elements)

    def supprimer(self, position: int) -> Any:  # noqa: ANN401 -- l'element retire
        """Retire l'element de ce rang ; les suivants avancent d'une case."""
        if not indice_valide(position, self.taille):
            msg = f"position hors du tableau : {position} pour {self.taille} elements"
            raise IndexError(msg)
        retire = self.elements[position]
        for rang in range(position, self.taille - 1):
            self.elements[rang] = self.elements[rang + 1]
        self.elements.pop()
        return retire

    def inserer(self, position: int, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """Ecrit a ce rang, apres avoir recule les suivants en partant de la fin."""
        if self.taille == self.capacite:
            msg = "tableau plein : il ne reste aucune case libre"
            raise OverflowError(msg)
        if not indice_valide(position, self.taille + 1):
            msg = f"position hors du tableau : {position} pour {self.taille} elements"
            raise IndexError(msg)
        self.elements.append(valeur)
        for rang in range(self.taille - 1, position, -1):
            self.elements[rang] = self.elements[rang - 1]
        self.elements[position] = valeur

    def ajouter_a_la_fin(self, valeur: Any) -> None:  # noqa: ANN401 -- tout type
        """Ecrit dans la premiere case libre."""
        self.inserer(self.taille, valeur)


def boucle_sur_les_elements(
    tableau: list[Any], effet: Callable[[Any], Any], *, par_reference: bool
) -> tuple[Any, ...]:
    """`for (auto elem : tab) { ...; elem = effet(elem); }` : ce que la boucle a vu.

    Par reference (`auto&`), l'affectation atteint le tableau ; sinon elle
    se perd avec la copie, a la fin de chaque tour.
    """
    vus = tuple(tableau)
    if par_reference:
        tableau[:] = [effet(element) for element in vus]
    return vus
