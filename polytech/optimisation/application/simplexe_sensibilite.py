"""Exercice IV du jeu « algorithme simplexe » : l'etude de sensibilite.

`min x - 2y + 6z` sous trois contraintes. Jusqu'ou un cout, ou un second
membre, peut-il varier sans que la base optimale change ?
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import sympy as sp

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import FormeStandard, forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.interpretation_duale import prix_fictifs
from polytech.optimisation.domain.tableau import tableau_de_la_base

PROGRAMME: Final = ProgrammeGeneral(
    Optimisation.MIN,
    (1, -2, 6),
    (
        Contrainte((1, 1, 2), Sens.INFERIEUR, 5),
        Contrainte((1, 0, 0), Sens.SUPERIEUR, 2),
        Contrainte((0, 1, 0), Sens.INFERIEUR, 4),
    ),
)
BASE: Final = (2, 1, 6)
"""Planche 36 : `y`, `x` et `x3e` sont en base a l'optimum."""
DELTA: Final = sp.Symbol("Delta", real=True)
Intervalle = tuple[sp.Expr, sp.Expr]


def optimum() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """La valeur et le point optimaux."""
    return optimum_fini(PROGRAMME)


def variables_duales() -> tuple[sp.Expr, ...]:
    """Les prix des trois contraintes."""
    return prix_fictifs(PROGRAMME)


def _intervalle(conditions: list[sp.Expr], centre: sp.Expr) -> Intervalle:
    """Les `Delta` qui rendent toutes les conditions positives, rapportes au centre."""
    permis = sp.S.Reals
    for condition in conditions:
        if condition.has(DELTA):
            permis = permis.intersect(
                sp.solve_univariate_inequality(condition >= 0, DELTA, relational=False)
            )
    return permis.inf + centre, permis.sup + centre


def intervalles_des_couts() -> tuple[Intervalle, ...]:
    """Pour chaque cout : les valeurs qui laissent les couts reduits negatifs."""
    standard = forme_standard(PROGRAMME)
    intervalles = []
    for j, cout in enumerate(PROGRAMME.objectif):
        couts = standard.couts.copy()
        couts[0, j] = cout + DELTA
        tableau = tableau_de_la_base(replace(standard, couts=couts), BASE)
        reduits = [-a for a in tableau.coefficients[0, 1:]]
        intervalles.append(_intervalle(reduits, cout))
    return tuple(intervalles)


def _solution_de_base(standard: FormeStandard) -> sp.Matrix:
    return tableau_de_la_base(standard, BASE).coefficients[1:, 0]


def intervalles_des_seconds_membres() -> tuple[Intervalle, ...]:
    """Pour chaque second membre : les valeurs qui laissent la base admissible."""
    standard = forme_standard(PROGRAMME)
    intervalles = []
    for i, contrainte in enumerate(PROGRAMME.contraintes):
        seconds = standard.seconds_membres.copy()
        seconds[i, 0] = contrainte.second_membre + DELTA
        base = _solution_de_base(replace(standard, seconds_membres=seconds))
        intervalles.append(_intervalle(list(base), contrainte.second_membre))
    return tuple(intervalles)


def la_base_tient(rang: int, *, dedans: bool) -> bool:
    """Vrai si la base reste admissible quand ce second membre est deplace.

    `dedans` le place une unite au-dessus de la borne basse de son
    intervalle, sinon une unite en dessous.
    """
    basse, _ = intervalles_des_seconds_membres()[rang]
    valeur = basse + (1 if dedans else -1)
    contraintes = list(PROGRAMME.contraintes)
    contraintes[rang] = replace(contraintes[rang], second_membre=valeur)
    deplace = replace(PROGRAMME, contraintes=tuple(contraintes))
    base = _solution_de_base(forme_standard(deplace))
    if any(v < 0 for v in base):
        return False
    couts = forme_standard(deplace).couts
    par_la_base = sum(couts[0, j - 1] * v for j, v in zip(BASE, base, strict=True))
    return bool(optimum_fini(deplace)[0] == par_la_base)
