"""Les trois tris en `N^2`, et le tableau des complexites.

Le cours (partie 1, planches 64 a 67) : un algorithme de tri range les
elements d'un tableau selon une relation d'ordre. Il en nomme sept, avec
leur complexite, et en ecrit deux :

* le **tri a bulles** compare chaque element a son voisin et les permute
  s'ils sont en desordre ; il recommence tant qu'une passe a permute ;
* le **tri par selection** cherche le plus petit des elements restants et
  le met a sa place : `N (N - 1) / 2` comparaisons, toujours, mais au plus
  `N - 1` permutations.

Le **tri par insertion** insere chaque element a sa place parmi ceux deja
tries : il ne coute que `N - 1` comparaisons sur un tableau trie.

Chaque fonction rend le tableau trie et ce qu'il a coute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

COMPLEXITES: Final = {
    "tri a bulles": "O(N^2)",
    "tri par selection": "O(N^2)",
    "tri par insertion": "O(N^2)",
    "tri de Shell": "O(N log^2 N)",
    "tri fusion": "O(N log N)",
    "tri rapide": "O(N log N)",
    "tri par tas": "O(N log N)",
}
"""Le tableau de la planche 64."""


@dataclass(frozen=True, slots=True)
class Tri:
    """Les valeurs triees, et le cout du tri."""

    valeurs: tuple[Any, ...]
    comparaisons: int
    permutations: int = 0
    passes: int = 0


def tri_a_bulles(valeurs: Sequence[Any]) -> Tri:
    """Des passes successives, jusqu'a une passe sans permutation."""
    x, comparaisons, permutations, passes = list(valeurs), 0, 0, 0
    permute = True
    while permute:
        permute, passes = False, passes + 1
        for i in range(len(x) - 1):
            comparaisons += 1
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                permute, permutations = True, permutations + 1
    return Tri(tuple(x), comparaisons, permutations, passes)


def tri_par_selection(valeurs: Sequence[Any]) -> Tri:
    """Le plus petit des restants, mis a sa place, rang apres rang."""
    x, comparaisons, permutations = list(valeurs), 0, 0
    for i in range(len(x) - 1):
        minimum = i
        for j in range(i + 1, len(x)):
            comparaisons += 1
            if x[j] < x[minimum]:
                minimum = j
        if minimum != i:
            x[i], x[minimum] = x[minimum], x[i]
            permutations += 1
    return Tri(tuple(x), comparaisons, permutations)


def tri_par_insertion(valeurs: Sequence[Any]) -> Tri:
    """Chaque element, insere a sa place parmi ceux deja tries."""
    x, comparaisons, permutations = list(valeurs), 0, 0
    for i in range(1, len(x)):
        j = i
        while j > 0:
            comparaisons += 1
            if x[j - 1] <= x[j]:
                break
            x[j - 1], x[j] = x[j], x[j - 1]
            permutations, j = permutations + 1, j - 1
    return Tri(tuple(x), comparaisons, permutations)
