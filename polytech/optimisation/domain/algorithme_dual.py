"""Chapitre V.2.2 : l'algorithme dual simplexe (phase II).

La ou l'algorithme simplexe va de base primale admissible en base primale
admissible jusqu'a la premiere qui soit aussi duale admissible,
l'algorithme dual fait l'inverse : il part d'une base **duale
admissible** et la garde telle, jusqu'a la premiere qui soit aussi primale
admissible -- donc optimale. C'est l'algorithme simplexe applique au
probleme dual, mais calcule sur le tableau du primal.

Les solutions intermediaires ne sont pas admissibles pour le primal, et
`z` s'y **degrade** : c'est `w` qui s'ameliore. Les formules de
changement de base sont celles du chapitre II, avec un pivot negatif.

Deux arrets : la solution optimale, ou le probleme primal impossible
(dual non borne). Un primal non borne est exclu : il n'aurait pas de base
duale admissible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.algorithme import (
    ITERATIONS_MAXIMALES,
    Issue,
    Regle,
    Resolution,
)
from polytech.optimisation.domain.criteres_duaux import (
    est_dual_admissible,
    est_primal_admissible,
    ligne_sans_solution_admissible,
    rapports_duaux,
    variable_entrante_duale,
    variable_sortante_duale,
)
from polytech.optimisation.domain.grand_m import ordre
from polytech.optimisation.domain.pivot import changer_de_base

if TYPE_CHECKING:
    from polytech.optimisation.domain.contraintes import Optimisation
    from polytech.optimisation.domain.tableau import TableauSimplexe


def _pivot_de_bland(tableau: TableauSimplexe) -> tuple[int, int]:
    """V.5.1 : `l = min{i | a_i0 < 0}`, `k = min{p | rapport minimal}`."""
    lignes = zip(tableau.base, tableau.coefficients[1:, 0], strict=True)
    sortante = min(i for i, valeur in lignes if ordre(valeur) < (0, 0))
    rapports = rapports_duaux(tableau, sortante)
    minimal = min(rapports.values())
    return sortante, min(j for j, rapport in rapports.items() if rapport == minimal)


def pivoter_en_dual(
    tableau: TableauSimplexe,
    regle: Regle = Regle.PLUS_FORT_COEFFICIENT,
) -> TableauSimplexe:
    """Une iteration : la variable negative sort, celle du plus petit rapport entre."""
    if regle is Regle.BLAND:
        sortante, entrante = _pivot_de_bland(tableau)
    else:
        sortante = variable_sortante_duale(tableau)
        entrante = variable_entrante_duale(tableau, sortante)
    return changer_de_base(tableau, sortante, entrante)


def phase_deux_duale(
    depart: TableauSimplexe,
    sens: Optimisation,
    regle: Regle = Regle.PLUS_FORT_COEFFICIENT,
    iterations_maximales: int = ITERATIONS_MAXIMALES,
) -> Resolution:
    """Deroule l'algorithme dual depuis une base duale admissible."""
    if not est_dual_admissible(depart, sens):
        msg = f"base non duale admissible : {depart.base} ; voir la phase I (V.3)"
        raise ValueError(msg)
    tableaux = [depart]
    for _ in range(iterations_maximales):
        courant = tableaux[-1]
        if est_primal_admissible(courant):
            return Resolution(tuple(tableaux), Issue.OPTIMALE)
        if ligne_sans_solution_admissible(courant) is not None:
            return Resolution(tuple(tableaux), Issue.IMPOSSIBLE)
        tableaux.append(pivoter_en_dual(courant, regle))
    msg = f"aucun arret apres {iterations_maximales} iterations duales : cyclage ?"
    raise RuntimeError(msg)
