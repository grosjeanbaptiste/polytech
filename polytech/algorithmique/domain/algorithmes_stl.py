"""Les algorithmes de la STL, sur une plage `[debut, fin)`.

Le cours (seances 5 et 6, « Algorithmes ») : « des patrons de fonctions
utilises pour faire des traitements sur les elements qui se retrouvent
dans les conteneurs ». Ils portent ici leur nom de `<algorithm>`, et une
position entiere y joue le role de l'iterateur : 0 pour `begin()`, la
taille pour `end()`.

Deux d'entre eux surprennent, et le cours y insiste :

* `remove` et `unique` **ne changent pas la taille** du conteneur. Ils
  ramenent vers l'avant les elements gardes et rendent la « nouvelle fin
  logique » ; ce qui suit est du reliquat. Retirer vraiment demande
  `erase(remove(...), end())` ;
* `unique` ne retire que les voisins egaux : il faut trier d'abord.

Une recherche infructueuse rend `end()`, c'est-a-dire la taille.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def remove_if(elements: list[Any], predicat: Callable[[Any], bool]) -> int:
    """Ramene vers l'avant ce qui ne satisfait pas le predicat ; rend la fin logique."""
    fin = 0
    for element in list(elements):
        if not predicat(element):
            elements[fin] = element
            fin += 1
    return fin


def remove(elements: list[Any], valeur: Any) -> int:  # noqa: ANN401 -- tout type
    """`remove(begin, end, valeur)` : la fin logique, sans changer la taille."""
    return remove_if(elements, lambda element: bool(element == valeur))


def unique(elements: list[Any]) -> int:
    """Ne garde que le premier de chaque groupe de voisins egaux ; la fin logique."""
    fin = 0
    for element in list(elements):
        if fin == 0 or elements[fin - 1] != element:
            elements[fin] = element
            fin += 1
    return fin


def sort(
    elements: list[Any],
    debut: int = 0,
    fin: int | None = None,
    *,
    decroissant: bool = False,
) -> None:
    """`sort(debut, fin)` : range la plage, et elle seule."""
    fin = len(elements) if fin is None else fin
    if not 0 <= debut <= fin <= len(elements):
        msg = f"plage hors du tableau : {debut} a {fin}"
        raise IndexError(msg)
    elements[debut:fin] = sorted(elements[debut:fin], reverse=decroissant)


def search(elements: Sequence[Any], cherchee: Sequence[Any]) -> int:
    """La position de la premiere occurrence de la sous-suite, ou `end()`."""
    largeur = len(cherchee)
    for position in range(len(elements) - largeur + 1):
        if tuple(elements[position : position + largeur]) == tuple(cherchee):
            return position
    return len(elements)


def find(elements: Sequence[Any], valeur: Any) -> int:  # noqa: ANN401 -- tout type
    """La position de la premiere occurrence de la valeur, ou `end()`."""
    return search(elements, (valeur,))


def count_if(elements: Sequence[Any], predicat: Callable[[Any], bool]) -> int:
    """Le nombre d'elements qui satisfont le predicat."""
    return sum(1 for element in elements if predicat(element))


def count(elements: Sequence[Any], valeur: Any) -> int:  # noqa: ANN401 -- tout type
    """Le nombre d'occurrences de la valeur."""
    return count_if(elements, lambda element: bool(element == valeur))


def all_of(elements: Sequence[Any], predicat: Callable[[Any], bool]) -> bool:
    """Vrai si tous les elements satisfont le predicat."""
    return count_if(elements, predicat) == len(elements)


def equal(premiere: Sequence[Any], seconde: Sequence[Any]) -> bool:
    """Vrai si les deux plages ont la meme taille et les memes elements."""
    return tuple(premiere) == tuple(seconde)


def replace(elements: list[Any], ancienne: Any, nouvelle: Any) -> None:  # noqa: ANN401 -- tout type
    """Remplace chaque occurrence de l'ancienne valeur par la nouvelle."""
    elements[:] = [nouvelle if e == ancienne else e for e in elements]


def copy(source: Sequence[Any], destination: list[Any], position: int = 0) -> int:
    """Copie la source a partir de la position ; rend la position qui suit."""
    fin = position + len(source)
    if fin > len(destination):
        msg = f"destination trop petite : {len(destination)} pour {fin}"
        raise IndexError(msg)
    destination[position:fin] = source
    return fin


def _extreme(elements: Sequence[Any], *, maximum: bool) -> int:
    if not elements:
        msg = "plage vide : ni minimum ni maximum"
        raise ValueError(msg)
    extreme = max(elements) if maximum else min(elements)
    return find(elements, extreme)


def min_element(elements: Sequence[Any]) -> int:
    """La position du premier des plus petits elements."""
    return _extreme(elements, maximum=False)


def max_element(elements: Sequence[Any]) -> int:
    """La position du premier des plus grands elements."""
    return _extreme(elements, maximum=True)


def reverse(elements: list[Any]) -> None:
    """Inverse l'ordre des elements, sur place."""
    elements.reverse()
