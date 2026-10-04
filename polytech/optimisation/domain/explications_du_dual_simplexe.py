"""Une iteration de l'algorithme dual, dite comme le cours la dit page 117.

`algorithme_dual.phase_deux_duale` rend les tableaux ; ce module rend ce
que le cours ecrit en regard : quelle variable sort et **pourquoi** (la
plus negative), laquelle entre et **pourquoi** (le plus petit rapport sur
les pivots negatifs), et aux deux arrets le cas du paragraphe V.1 qui les
justifie. L'ordre est l'inverse de celui du simplexe : on choisit d'abord
la ligne, ensuite la colonne.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.algorithme_dual import phase_deux_duale
from polytech.optimisation.domain.arrets_duaux_expliques import (
    QUESTION,
    SOURCE,
    expliquer_l_arret_impossible,
    expliquer_l_arret_optimal,
)
from polytech.optimisation.domain.criteres_duaux import (
    est_primal_admissible,
    ligne_sans_solution_admissible,
    rapports_duaux,
    variable_entrante_duale,
    variable_sortante_duale,
)
from polytech.optimisation.domain.pivot import changer_de_base
from polytech.raisonnement.domain.etapes import Etape, Raisonnement

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.contraintes import Optimisation
    from polytech.optimisation.domain.tableau import TableauSimplexe


def _rapport(numerateur: sp.Expr, denominateur: sp.Expr) -> str:
    bas = str(denominateur) if denominateur.is_integer else f"({denominateur})"
    return f"{numerateur}/{bas}"


def _sortie(tableau: TableauSimplexe, sortante: int) -> Etape:
    valeurs = [v for v in tableau.coefficients[1:, 0] if v < 0]
    ligne = tableau.base.index(sortante) + 1
    justification = (
        "seule variable de base negative"
        if len(valeurs) == 1
        else f"{tableau.coefficients[ligne, 0]} = min({', '.join(map(str, valeurs))})"
    )
    source = f"{SOURCE}, V.2.1, p. 116"
    return Etape(f"x_{sortante} sort de la base", justification, source)


def _entree(tableau: TableauSimplexe, sortante: int, entrante: int) -> Etape:
    ligne = tableau.base.index(sortante) + 1
    candidats = [
        (tableau.coefficients[0, tableau.hors_base.index(j) + 1], j)
        for j in rapports_duaux(tableau, sortante)
    ]
    pivots = {
        j: tableau.coefficients[ligne, tableau.hors_base.index(j) + 1]
        for _, j in candidats
    }
    if len(candidats) == 1:
        justification = f"{pivots[entrante]} est le seul pivot negatif de la ligne"
    else:
        rapports = {j: _rapport(a, pivots[j]) for a, j in candidats}
        justification = (
            f"{rapports[entrante]} = min({', '.join(rapports.values())}) sur les "
            "a_lj negatifs"
        )
    source = f"{SOURCE}, V.1, cas 2.b, p. 115"
    return Etape(f"x_{entrante} entre dans la base", justification, source)


def _changement(tableau: TableauSimplexe) -> Raisonnement:
    sortante = variable_sortante_duale(tableau)
    entrante = variable_entrante_duale(tableau, sortante)
    ligne = tableau.base.index(sortante) + 1
    colonne = tableau.hors_base.index(entrante) + 1
    pivot = Etape(
        f"le pivot est a_{sortante}{entrante} = {tableau.coefficients[ligne, colonne]}",
        "c'est le coefficient, negatif, a l'intersection de la ligne sortante et "
        "de la colonne entrante",
        f"{SOURCE}, V.2.3, remarque 2, p. 117",
    )
    nouvelle = changer_de_base(tableau, sortante, entrante).valeur
    etapes = (_sortie(tableau, sortante), _entree(tableau, sortante, entrante), pivot)
    conclusion = (
        f"z passe de {tableau.valeur} a {nouvelle} : il se degrade, w s'ameliore"
    )
    return Raisonnement(QUESTION, etapes, conclusion)


def expliquer_une_iteration_duale(tableau: TableauSimplexe) -> Raisonnement:
    """Ce que l'algorithme dual fait de ce tableau, et le cas qui le justifie."""
    if est_primal_admissible(tableau):
        return expliquer_l_arret_optimal(tableau)
    bloquee = ligne_sans_solution_admissible(tableau)
    if bloquee is not None:
        return expliquer_l_arret_impossible(tableau, bloquee)
    return _changement(tableau)


def expliquer_la_resolution_duale(
    depart: TableauSimplexe,
    sens: Optimisation,
) -> Raisonnement:
    """Toutes les iterations enchainees, du tableau de depart a l'arret."""
    tableaux = phase_deux_duale(depart, sens).tableaux
    raisonnement = expliquer_une_iteration_duale(tableaux[0])
    for tableau in tableaux[1:]:
        raisonnement = raisonnement.puis(expliquer_une_iteration_duale(tableau))
    return raisonnement


__all__ = ["SOURCE", "expliquer_la_resolution_duale", "expliquer_une_iteration_duale"]
