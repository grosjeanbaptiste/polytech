"""Les algorithmes de la STL qui supposent une sequence triee.

Le syllabus de programmation objet (4.4) :

* `lower_bound` et `upper_bound` : « recherche binaire determinant la
  premiere et la derniere position a laquelle Val peut etre inseree dans la
  sequence ordonnee [...] sans en briser l'ordre » ;
* `merge` : « fusionner deux listes d'elements ordonnees ».

Tous trois n'ont de sens que sur des sequences triees. En C++ le resultat
est alors indefini ; ici c'est une erreur.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _exiger_triee(elements: Sequence[Any]) -> None:
    if any(a > b for a, b in pairwise(elements)):
        msg = "sequence non triee : la recherche binaire et la fusion l'exigent"
        raise ValueError(msg)


def lower_bound(elements: Sequence[Any], valeur: Any) -> int:  # noqa: ANN401 -- tout type ordonne
    """La premiere position ou inserer la valeur sans briser l'ordre."""
    _exiger_triee(elements)
    return bisect_left(elements, valeur)


def upper_bound(elements: Sequence[Any], valeur: Any) -> int:  # noqa: ANN401 -- tout type ordonne
    """La derniere position ou inserer la valeur sans briser l'ordre."""
    _exiger_triee(elements)
    return bisect_right(elements, valeur)


def merge(premiere: Sequence[Any], seconde: Sequence[Any]) -> list[Any]:
    """La fusion de deux sequences triees ; a egalite, la premiere passe d'abord."""
    _exiger_triee(premiere)
    _exiger_triee(seconde)
    fusion, i, j = [], 0, 0
    while i < len(premiere) and j < len(seconde):
        if seconde[j] < premiere[i]:
            fusion.append(seconde[j])
            j += 1
        else:
            fusion.append(premiere[i])
            i += 1
    return [*fusion, *premiere[i:], *seconde[j:]]
