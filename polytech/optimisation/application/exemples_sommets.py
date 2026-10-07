"""Exemples resolus, planches 8 a 30 et 41 a 43 : bases et sommets.

Une base est un choix de `m` colonnes regulieres de la matrice `T` ; sa
solution de base est un sommet du domaine si elle est positive. Le
simplexe passe de sommet en sommet.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.tableau import TableauSimplexe, tableau_de_la_base

if TYPE_CHECKING:
    import sympy as sp

INF: Final = Sens.INFERIEUR
Base = tuple[int, ...]


def _programme(
    sens: Optimisation, objectif: Base, *lignes: tuple[Base, int]
) -> ProgrammeGeneral:
    contraintes = tuple(Contrainte(a, INF, d) for a, d in lignes)
    return ProgrammeGeneral(sens, objectif, contraintes)


PREMIER: Final = _programme(
    Optimisation.MAX, (6, 5), ((1, 1), 8), ((-2, 3), 6), ((1, -1), 2)
)
"""Planches 8 a 11 et 24 a 30 : l'exemple I."""
SECOND: Final = _programme(
    Optimisation.MAX, (2, 5), ((1, 0), 4), ((0, 1), 3), ((1, 2), 8)
)
"""Planches 12 et 13."""
DEGENERE: Final = _programme(
    Optimisation.MIN, (-1, 1), ((2, -1), 4), ((1, -2), 2), ((1, 1), 5)
)
"""Planches 41 et 42."""
ARETE: Final = _programme(
    Optimisation.MIN, (-5, -5), ((1, 1), 8), ((-2, 3), 6), ((1, -1), 2)
)
"""Planche 43."""
PARCOURS: Final = ((3, 4, 5), (3, 4, 1), (2, 4, 1))
"""Les bases des trois tableaux de l'exemple I (planches 27 a 29)."""


def _tableaux(programme: ProgrammeGeneral) -> dict[Base, TableauSimplexe]:
    """Le tableau de chaque ensemble de colonnes qui forme une base."""
    standard = forme_standard(programme)
    equations, colonnes = standard.matrice.shape
    tableaux = {}
    for base in itertools.combinations(range(1, colonnes + 1), equations):
        if standard.matrice[:, [i - 1 for i in base]].det() != 0:
            tableaux[base] = tableau_de_la_base(standard, base)
    return tableaux


def bases_regulieres(programme: ProgrammeGeneral) -> int:
    """Le nombre d'ensembles de colonnes qui forment une base."""
    return len(_tableaux(programme))


def ensembles_singuliers(programme: ProgrammeGeneral) -> tuple[Base, ...]:
    """Les ensembles de colonnes dont la matrice n'est pas reguliere."""
    standard = forme_standard(programme)
    equations, colonnes = standard.matrice.shape
    tous = itertools.combinations(range(1, colonnes + 1), equations)
    return tuple(base for base in tous if base not in _tableaux(programme))


def solution_de_base(programme: ProgrammeGeneral, base: Base) -> dict[int, sp.Expr]:
    """La valeur de chaque variable de base ; les autres sont nulles."""
    tableau = tableau_de_la_base(forme_standard(programme), base)
    return dict(zip(base, tableau.coefficients[1:, 0], strict=True))


def est_un_sommet(programme: ProgrammeGeneral, base: Base) -> bool:
    """Vrai si la solution de base est positive, donc admissible."""
    return all(v >= 0 for v in solution_de_base(programme, base).values())


def _point(tableau: TableauSimplexe, variables: int) -> tuple[sp.Expr, ...]:
    return tuple(tableau.solution(variables)[:2])


def bases_du_sommet(programme: ProgrammeGeneral, sommet: Base) -> tuple[Base, ...]:
    """Toutes les bases admissibles dont la solution est ce sommet."""
    variables = forme_standard(programme).matrice.cols
    return tuple(
        base
        for base, tableau in _tableaux(programme).items()
        if est_un_sommet(programme, base) and _point(tableau, variables) == sommet
    )


def sommets_optimaux(
    programme: ProgrammeGeneral,
) -> tuple[sp.Expr, tuple[tuple[sp.Expr, ...], ...]]:
    """La meilleure valeur parmi les sommets, et les sommets qui l'atteignent."""
    variables = forme_standard(programme).matrice.cols
    admissibles = [
        (tableau.valeur, _point(tableau, variables))
        for base, tableau in _tableaux(programme).items()
        if est_un_sommet(programme, base)
    ]
    minimum = min(valeur for valeur, _ in admissibles)
    gagnants = sorted({p for valeur, p in admissibles if valeur == minimum})
    return minimum, tuple(gagnants)


def parcours() -> tuple[tuple[tuple[sp.Expr, ...], sp.Expr], ...]:
    """Le point et la valeur de chacun des trois tableaux de l'exemple I."""
    standard = forme_standard(PREMIER)
    tableaux = [tableau_de_la_base(standard, base) for base in PARCOURS]
    return tuple((_point(t, standard.matrice.cols), t.valeur) for t in tableaux)


def couts_reduits_finaux() -> dict[int, sp.Expr]:
    """La premiere ligne du tableau optimal de l'exemple I, par variable hors base."""
    final = tableau_de_la_base(forme_standard(PREMIER), PARCOURS[-1])
    return dict(zip(final.hors_base, final.coefficients[0, 1:], strict=True))
