"""Chapitre XIV.3.3 : les penalites, et les regles de selection qu'elles fondent.

Quelle variable separer, et dans quelle branche descendre d'abord ? Les
**penalites** `P_l^(a)` et `P_l^(b)` minorent la degradation de `z` dans
chacune des deux branches : c'est l'augmentation de la fonction
economique a la **premiere** iteration de la reoptimisation duale. En
minimisation (`a_0j <= 0`) :

    P_l^(a) = min sur a_lj > 0 de  -f a_0j / a_lj,
    P_l^(b) = min sur a_lj < 0 de  (1 - f) a_0j / a_lj,

infinie quand aucun pivot n'existe : la branche est impossible.

Trois regles : descendre dans la branche de la **plus petite** penalite
(vers une bonne solution, vite) ; a l'oppose de la **plus grande** (pour
elaguer l'autre des que possible) ; la **regle 3**, la variable dont la
plus petite penalite est la plus grande. Le **maximum de
non-integralite** remplace les penalites par `f` et `1 - f`.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.separation_sur_tableau import (
    Branche,
    partie_decimale,
)

if TYPE_CHECKING:
    from polytech.optimisation.domain.tableau import TableauSimplexe

type Choix = tuple[int, Branche]


class Regle(Enum):
    """Comment choisir la variable de separation et la premiere branche."""

    PLUS_PETITE_PENALITE = "regle 1 : dans la branche de la plus petite penalite"
    PLUS_GRANDE_PENALITE = "regle 2 : a l'oppose de la plus grande penalite"
    REGLE_3 = "regle 3 : la plus petite penalite la plus grande"
    NON_INTEGRALITE = "maximum de non-integralite : f et 1 - f pour penalites"


def non_integralite(tableau: TableauSimplexe, indice: int) -> sp.Expr:
    """`min(f, 1 - f)` : a quel point `x_l` est loin d'un entier."""
    f = partie_decimale(tableau.coefficients[tableau.base.index(indice) + 1, 0])
    return sp.Min(f, 1 - f)


def penalites(tableau: TableauSimplexe, indice: int) -> tuple[sp.Expr, sp.Expr]:
    """`(P_l^(a), P_l^(b))` pour la variable de base `x_l`."""
    ligne = tableau.base.index(indice) + 1
    f = partie_decimale(tableau.coefficients[ligne, 0])
    couples = [
        (tableau.coefficients[0, j], tableau.coefficients[ligne, j])
        for j in range(1, tableau.coefficients.cols)
    ]
    basse = [-f * a0 / a for a0, a in couples if a > 0]
    haute = [(1 - f) * a0 / a for a0, a in couples if a < 0]
    return sp.Min(*basse) if basse else sp.oo, sp.Min(*haute) if haute else sp.oo


def _candidates(
    tableau: TableauSimplexe,
    entieres: frozenset[int],
    regle: Regle,
) -> dict[Choix, sp.Expr]:
    """La penalite de chaque couple (variable non entiere, branche)."""
    mesures: dict[Choix, sp.Expr] = {}
    for ligne, indice in enumerate(tableau.base, start=1):
        f = partie_decimale(tableau.coefficients[ligne, 0])
        if indice not in entieres or f == 0:
            continue
        basse, haute = (
            (f, 1 - f) if regle is Regle.NON_INTEGRALITE else penalites(tableau, indice)
        )
        mesures[indice, Branche.BASSE] = basse
        mesures[indice, Branche.HAUTE] = haute
    return mesures


def choisir_la_separation(
    tableau: TableauSimplexe,
    entieres: frozenset[int],
    regle: Regle,
) -> Choix:
    """La variable a separer et la branche a examiner la premiere."""
    mesures = _candidates(tableau, entieres, regle)
    if not mesures:
        msg = "aucune variable astreinte n'est fractionnaire : rien a separer"
        raise ValueError(msg)
    if regle is Regle.PLUS_PETITE_PENALITE:
        return min(mesures, key=lambda choix: mesures[choix])
    if regle is Regle.PLUS_GRANDE_PENALITE:
        variable, branche = max(mesures, key=lambda choix: mesures[choix])
        oppose = Branche.HAUTE if branche is Branche.BASSE else Branche.BASSE
        return variable, oppose
    variables = dict.fromkeys(variable for variable, _ in mesures)
    plus_petite = {
        v: (
            Branche.BASSE
            if mesures[v, Branche.BASSE] <= mesures[v, Branche.HAUTE]
            else Branche.HAUTE
        )
        for v in variables
    }
    retenue = max(variables, key=lambda v: mesures[v, plus_petite[v]])
    return retenue, plus_petite[retenue]
