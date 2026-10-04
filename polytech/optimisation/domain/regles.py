"""Chapitre II.6.3 : deux regles qui empechent le cyclage.

Quand deux variables peuvent sortir -- des rapports `a_i0 / a_ik` a
egalite --, la nouvelle solution de base est degeneree, et l'algorithme
peut revenir a une base deja visitee : c'est le **cyclage**.

**La regle lexicographique** traduit la methode de perturbation. Sur le
tableau **complet**, a egalite des rapports `a_i0 / a_ik`, comparer les
`a_i1 / a_ik`, puis les `a_i2 / a_ik`, et retenir le plus petit ; la
matrice identite garantit qu'un rang finit par departager.

**La regle de Bland** : faire entrer le plus petit indice `k` qui
ameliore `z`, faire sortir le plus petit indice `l` parmi ceux qui
realisent le plus petit rapport. Elle assure la convergence meme
degeneree, au prix d'un plus grand nombre d'iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres import variables_sortantes
from polytech.optimisation.domain.grand_m import ordre

if TYPE_CHECKING:
    from polytech.optimisation.domain.tableau import TableauSimplexe


def sortante_lexicographique(
    tableau: TableauSimplexe,
    entrante: int,
    variables: int,
) -> int:
    """Parmi les sortantes a egalite, celle du plus petit rapport lexicographique."""
    complet = tableau.complet(variables)
    candidates = variables_sortantes(tableau, entrante)

    def rapports(indice: int) -> tuple[object, ...]:
        ligne = tableau.base.index(indice) + 1
        pivot = complet[ligne, entrante]
        return tuple(complet[ligne, j] / pivot for j in range(1, variables + 1))

    return min(candidates, key=rapports)


def entrante_de_bland(tableau: TableauSimplexe, sens: Optimisation) -> int:
    """`k = min{j | a_0j ameliore z}` : le plus petit indice, non le plus fort."""
    signe = 1 if sens is Optimisation.MIN else -1
    ameliorants = [
        indice
        for rang, indice in enumerate(tableau.hors_base, start=1)
        if ordre(signe * tableau.coefficients[0, rang]) > (0, 0)
    ]
    if not ameliorants:
        msg = f"aucune variable entrante : le tableau est optimal ({tableau.base})"
        raise ValueError(msg)
    return min(ameliorants)


def sortante_de_bland(tableau: TableauSimplexe, entrante: int) -> int:
    """Le plus petit indice parmi ceux qui realisent le plus petit rapport."""
    return min(variables_sortantes(tableau, entrante))
