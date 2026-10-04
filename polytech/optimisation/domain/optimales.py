"""Chapitre II.7.1 : determiner toutes les solutions optimales.

Sur le tableau d'une solution de base optimale :

* si aucun `a_0j` hors base n'est nul, la solution optimale est
  **unique** ;
* si un `a_0k` est nul, augmenter `x_k` ne change pas `z`. Ou bien un
  pivot positif borne ce deplacement, et l'on atteint un **autre sommet
  optimal** -- tous les points de l'arete parcourue sont optimaux ; ou
  bien aucun ne le borne, et l'arete est un **rayon** de solutions
  optimales.

Si la solution est degeneree, un `a_0k` nul ne suffit pas a affirmer la
non-unicite : le pivot peut ramener a la meme solution sous une autre
base (page 71, point b).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres import variables_sortantes
from polytech.optimisation.domain.grand_m import ordre
from polytech.optimisation.domain.pivot import changer_de_base

if TYPE_CHECKING:
    from polytech.optimisation.domain.tableau import TableauSimplexe


def _indifferentes(tableau: TableauSimplexe) -> list[int]:
    """Les variables hors base dont le coefficient `a_0k` est nul."""
    return [
        indice
        for rang, indice in enumerate(tableau.hors_base, start=1)
        if tableau.coefficients[0, rang] == 0
    ]


def est_l_unique_optimum(tableau: TableauSimplexe, sens: Optimisation) -> bool:
    """Propriete II.7.1 : tous les `a_0j` sont strictement defavorables."""
    signe = 1 if sens is Optimisation.MIN else -1
    return all(ordre(signe * a) < (0, 0) for a in tableau.coefficients[0, 1:])


def autres_sommets_optimaux(tableau: TableauSimplexe) -> tuple[TableauSimplexe, ...]:
    """Les tableaux des sommets optimaux adjacents, un par `a_0k` nul et borne."""
    voisins = []
    for entrante in _indifferentes(tableau):
        sortantes = variables_sortantes(tableau, entrante)
        if sortantes:
            voisins.append(changer_de_base(tableau, sortantes[0], entrante))
    return tuple(voisins)


def colonnes_des_rayons_optimaux(tableau: TableauSimplexe) -> tuple[int, ...]:
    """Les `k` tels que `a_0k = 0` sans pivot positif : un rayon d'optimums."""
    return tuple(
        entrante
        for entrante in _indifferentes(tableau)
        if not variables_sortantes(tableau, entrante)
    )
