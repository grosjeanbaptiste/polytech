"""Chapitre II.7.2 : l'interpretation economique du tableau simplexe.

La premiere colonne donne la valeur de la fonction economique, `a_00`, et
les niveaux des activites de base, `a_i0`. Les autres disent ce qui se
passe si une activite `k` hors base passe de zero a **une unite** :

* les activites de base deviennent `x_i = a_i0 - a_ik` : `a_ik` est le
  **taux marginal** de l'activite `k` a l'activite `i` ;
* la fonction economique devient `z = a_00 - a_0k` : `a_0k` est le
  **cout marginal** de l'activite `k`, et `-a_0k` son cout reduit.

Quand `x_k` est la variable d'ecart d'une contrainte, `a_0k` est la
**valeur marginale** du bien correspondant -- la variable duale du
chapitre IV. Si la matrice `T` contient une matrice unite, les `a_0j` de
ses colonnes donnent donc la solution duale associee a la base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.criteres import variables_sortantes

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.tableau import TableauSimplexe


def _colonne(tableau: TableauSimplexe, indice: int) -> int:
    return tableau.hors_base.index(indice) + 1


def effet_d_une_unite(
    tableau: TableauSimplexe,
    activite: int,
) -> tuple[sp.Expr, dict[int, sp.Expr]]:
    """`z` et les niveaux de base quand l'activite hors base passe a une unite."""
    colonne = _colonne(tableau, activite)
    niveaux = {
        indice: tableau.coefficients[ligne, 0] - tableau.coefficients[ligne, colonne]
        for ligne, indice in enumerate(tableau.base, start=1)
    }
    return tableau.valeur - tableau.coefficients[0, colonne], niveaux


def valeurs_marginales(
    tableau: TableauSimplexe,
    colonnes_unite: tuple[int, ...],
) -> tuple[sp.Expr, ...]:
    """Les `a_0j` des colonnes unite de `T` ; zero pour une variable en base."""
    return tuple(
        tableau.coefficients[0, _colonne(tableau, j)] if j in tableau.hors_base else 0
        for j in colonnes_unite
    )


def gain_de_l_iteration(tableau: TableauSimplexe, entrante: int) -> sp.Expr:
    """`z* - a_00 = -a_0k a_l0 / a_lk` : ce que `z` gagne a faire entrer `x_k`."""
    colonne = _colonne(tableau, entrante)
    sortantes = variables_sortantes(tableau, entrante)
    if not sortantes:
        msg = f"aucun pivot positif pour x_{entrante} : z n'est pas borne sur ce rayon"
        raise ValueError(msg)
    sortante = sortantes[0]
    ligne = tableau.base.index(sortante) + 1
    rapport = tableau.coefficients[ligne, 0] / tableau.coefficients[ligne, colonne]
    return -tableau.coefficients[0, colonne] * rapport
