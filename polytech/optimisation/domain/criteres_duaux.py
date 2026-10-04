"""Chapitre V.1 : les tests de l'algorithme dual, lus sur le tableau simplexe.

La base est supposee **duale admissible** : `a_0j <= 0` hors base (en
minimisation ; `>= 0` en maximisation). Alors :

1. si `a_i0 >= 0` pour tout `i`, elle est aussi primale admissible, donc
   optimale ;
2. sinon une variable `x_l` negative sort de la base -- la plus negative,
   regle usuelle (V.2.1) -- et
   a) si sa ligne n'a aucun `a_lj < 0`, le dual n'est pas borne et le
      primal est **impossible** ;
   b) sinon entre la variable `x_k` du plus petit rapport `a_0k / a_lk`
      sur les `a_lj < 0` : c'est la plus grande valeur du parametre theta
      qui garde la solution duale admissible.

`M`, quand la contrainte artificielle l'introduit (V.3), est aussi grand
que l'on veut : les `a_i0` se comparent par `grand_m.ordre`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.criteres import est_optimal
from polytech.optimisation.domain.grand_m import ordre

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.contraintes import Optimisation
    from polytech.optimisation.domain.tableau import TableauSimplexe


def _negatives(tableau: TableauSimplexe) -> dict[int, sp.Expr]:
    """Les variables de base negatives, `M` etant aussi grand qu'on veut."""
    valeurs = zip(tableau.base, tableau.coefficients[1:, 0], strict=True)
    return {i: valeur for i, valeur in valeurs if ordre(valeur) < (0, 0)}


def est_primal_admissible(tableau: TableauSimplexe) -> bool:
    """Condition 1 : `a_i0 >= 0` pour toute variable de base."""
    return not _negatives(tableau)


def est_dual_admissible(tableau: TableauSimplexe, sens: Optimisation) -> bool:
    """Condition 2 : `a_0j <= 0` hors base (`>= 0` en maximisation)."""
    return est_optimal(tableau, sens)


def est_dual_degenere(tableau: TableauSimplexe) -> bool:
    """V.5.1 : un `a_0j` nul hors base -- une contrainte duale serree de trop."""
    return any(a == 0 for a in tableau.coefficients[0, 1:])


def variable_sortante_duale(tableau: TableauSimplexe) -> int:
    """`a_l0 = min a_i0 < 0` : la variable de base la plus negative."""
    negatives = _negatives(tableau)
    if not negatives:
        msg = f"aucune variable de base negative : base {tableau.base} admissible"
        raise ValueError(msg)
    return min(negatives, key=lambda i: ordre(negatives[i]))


def rapports_duaux(tableau: TableauSimplexe, sortante: int) -> dict[int, sp.Expr]:
    """`|a_0j / a_lj|` pour chaque `j` hors base dont `a_lj < 0`."""
    ligne = tableau.base.index(sortante) + 1
    return {
        j: abs(tableau.coefficients[0, rang] / tableau.coefficients[ligne, rang])
        for rang, j in enumerate(tableau.hors_base, start=1)
        if tableau.coefficients[ligne, rang] < 0
    }


def ligne_sans_solution_admissible(tableau: TableauSimplexe) -> int | None:
    """Cas 2.a : une variable negative dont la ligne n'a aucun pivot negatif."""
    for indice in _negatives(tableau):
        if not rapports_duaux(tableau, indice):
            return indice
    return None


def variable_entrante_duale(tableau: TableauSimplexe, sortante: int) -> int:
    """Cas 2.b : le plus petit rapport ; a egalite, la regle lexicographique.

    Le rapport se prend en valeur absolue : la regle est alors la meme en
    minimisation et en maximisation (V.2.3, remarque 1).
    """
    rapports = rapports_duaux(tableau, sortante)
    if not rapports:
        msg = f"aucun pivot negatif sur la ligne de x_{sortante} : primal impossible"
        raise ValueError(msg)
    minimal = min(rapports.values())
    candidates = [j for j, rapport in rapports.items() if rapport == minimal]
    return min(candidates, key=lambda j: _colonne_reduite(tableau, sortante, j))


def _colonne_reduite(
    tableau: TableauSimplexe,
    sortante: int,
    j: int,
) -> tuple[sp.Expr, ...]:
    """`|a_1j / a_lj|, ..., |a_nj / a_lj|`, le tableau complete des `x_j = x_j`."""
    ligne = tableau.base.index(sortante) + 1
    rang = tableau.hors_base.index(j) + 1
    pivot = tableau.coefficients[ligne, rang]
    variables = max((*tableau.base, *tableau.hors_base))
    colonne = []
    for indice in range(1, variables + 1):
        if indice in tableau.base:
            valeur = tableau.coefficients[tableau.base.index(indice) + 1, rang]
        else:
            valeur = 1 if indice == j else 0
        colonne.append(abs(valeur / pivot))
    return tuple(colonne)
