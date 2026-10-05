"""L'arbre et le co-arbre d'un reseau maille.

Le syllabus d'electricite generale (13.17). Un reseau de `n` noeuds et
`b` branches n'a que `n - 1` equations de premiere espece independantes.
Pour ecrire a coup sur les autres, on trace un **arbre** : un chemin qui
joint tous les noeuds en suivant des branches, sans former de maille. Il
compte `n - 1` branches ; celles qui restent forment le co-arbre.

Chaque branche du co-arbre -- un maillon -- se referme par l'arbre d'une
seule maniere : c'est sa maille, et les `b - (n - 1)` mailles ainsi
obtenues donnent des equations de deuxieme espece independantes, chacune
portant une branche qu'aucune autre ne contient.

Un graphe est ici la suite de ses branches, chacune par ses deux noeuds ;
une maille est une suite de pas `(rang, sens)`, le sens valant `+1` quand
la branche est parcourue de son premier noeud vers son second.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    Graphe = Sequence[tuple[str, str]]
    Pas = tuple[int, int]


def nombre_de_mailles_independantes(*, noeuds: int, branches: int) -> int:
    """`b - (n - 1)` : le nombre de branches du co-arbre."""
    if branches < noeuds - 1:
        msg = f"trop peu de branches : {branches} pour relier {noeuds} noeuds"
        raise ValueError(msg)
    return branches - (noeuds - 1)


def _atteints(
    graphe: Graphe, arbre: Collection[int], depart: str
) -> dict[str, tuple[Pas, ...]]:
    """Le chemin, par les branches de l'arbre, vers chaque noeud atteint."""
    chemins: dict[str, tuple[Pas, ...]] = {depart: ()}
    file = [depart]
    while file:
        noeud = file.pop()
        for rang in arbre:
            premier, second = graphe[rang]
            for ici, la, sens in ((premier, second, 1), (second, premier, -1)):
                if ici == noeud and la not in chemins:
                    chemins[la] = (*chemins[noeud], (rang, sens))
                    file.append(la)
    return chemins


def est_un_arbre(graphe: Graphe, arbre: Collection[int]) -> bool:
    """Vrai si ces `n - 1` branches joignent tous les noeuds du graphe."""
    noeuds = {noeud for branche in graphe for noeud in branche}
    if len(arbre) != len(noeuds) - 1:
        return False
    return len(_atteints(graphe, arbre, min(noeuds))) == len(noeuds)


def maille_fondamentale(
    graphe: Graphe, arbre: Collection[int], maillon: int
) -> tuple[Pas, ...]:
    """La maille que ce maillon ferme par l'arbre, parcourue dans son sens."""
    if maillon in arbre:
        msg = f"la branche {maillon} appartient a l'arbre : elle ne ferme aucune maille"
        raise ValueError(msg)
    if not est_un_arbre(graphe, arbre):
        msg = f"ces branches ne forment pas un arbre du reseau : {sorted(arbre)}"
        raise ValueError(msg)
    premier, second = graphe[maillon]
    return ((maillon, 1), *_atteints(graphe, arbre, second)[premier])
