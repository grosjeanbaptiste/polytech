"""Les quatre tris que le cours donne pour plus rapides que `N^2`.

Le tableau de la planche 64 les nomme sans les ecrire :

* le **tri de Shell** est un tri par insertion sur des elements distants
  d'un ecart que l'on divise par deux jusqu'a 1 ;
* le **tri fusion** coupe le tableau en deux, trie chaque moitie, puis les
  fusionne : `N log N` comparaisons, quel que soit le tableau ;
* le **tri rapide** range les elements de part et d'autre d'un pivot, puis
  trie chaque cote. Il est en `N log N` **en moyenne** ; avec le premier
  element pour pivot, un tableau deja trie le ramene a `N (N - 1) / 2` ;
* le **tri par tas** range le tableau en tas, puis en retire le plus grand
  element, `N` fois.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polytech.algorithmique.domain.tris import Tri

if TYPE_CHECKING:
    from collections.abc import Sequence


def tri_de_shell(valeurs: Sequence[Any]) -> Tri:
    """Un tri par insertion a ecart decroissant : `N/2`, `N/4`, ..., 1."""
    x, comparaisons, ecart = list(valeurs), 0, len(valeurs) // 2
    while ecart > 0:
        for i in range(ecart, len(x)):
            j = i
            while j >= ecart:
                comparaisons += 1
                if x[j - ecart] <= x[j]:
                    break
                x[j - ecart], x[j] = x[j], x[j - ecart]
                j -= ecart
        ecart //= 2
    return Tri(tuple(x), comparaisons)


def tri_fusion(valeurs: Sequence[Any]) -> Tri:
    """Deux moities triees, puis fusionnees."""
    if len(valeurs) <= 1:
        return Tri(tuple(valeurs), 0)
    gauche = tri_fusion(valeurs[: len(valeurs) // 2])
    droite = tri_fusion(valeurs[len(valeurs) // 2 :])
    a, b, fusion = list(gauche.valeurs), list(droite.valeurs), []
    comparaisons = gauche.comparaisons + droite.comparaisons
    while a and b:
        comparaisons += 1
        fusion.append(a.pop(0) if a[0] <= b[0] else b.pop(0))
    return Tri(tuple(fusion + a + b), comparaisons)


def tri_rapide(valeurs: Sequence[Any]) -> Tri:
    """Le premier element pour pivot, les plus petits a gauche, les autres a droite."""
    if len(valeurs) <= 1:
        return Tri(tuple(valeurs), 0)
    pivot, reste = valeurs[0], valeurs[1:]
    petits = tri_rapide([v for v in reste if v < pivot])
    grands = tri_rapide([v for v in reste if v >= pivot])
    comparaisons = len(reste) + petits.comparaisons + grands.comparaisons
    return Tri((*petits.valeurs, pivot, *grands.valeurs), comparaisons)


def _tamiser(x: list[Any], racine: int, fin: int) -> int:
    """Descend la racine a sa place dans le tas ; rend le nombre de comparaisons."""
    comparaisons = 0
    while (fils := 2 * racine + 1) < fin:
        if fils + 1 < fin:
            comparaisons += 1
            fils += x[fils + 1] > x[fils]
        comparaisons += 1
        if x[racine] >= x[fils]:
            break
        x[racine], x[fils] = x[fils], x[racine]
        racine = fils
    return comparaisons


def tri_par_tas(valeurs: Sequence[Any]) -> Tri:
    """Le tableau range en tas, puis vide par son plus grand element."""
    x, comparaisons = list(valeurs), 0
    for racine in range(len(x) // 2 - 1, -1, -1):
        comparaisons += _tamiser(x, racine, len(x))
    for fin in range(len(x) - 1, 0, -1):
        x[0], x[fin] = x[fin], x[0]
        comparaisons += _tamiser(x, 0, fin)
    return Tri(tuple(x), comparaisons)
