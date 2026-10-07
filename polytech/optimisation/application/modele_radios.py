"""Modelisation, probleme V : 20000 radios en quatre semaines.

Quarante ouvriers formes produisent 50 radios par semaine chacun ; il en
faut davantage. Un formateur instruit trois eleves en une semaine. `F`
compte les formes, `T` ceux qui produisent, `I` ceux qui instruisent,
`E` les eleves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.modele_variables import Variables, numerotees
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

SEMAINES: Final = 4
A_LIVRER, CADENCE, PERMANENTS, ELEVES_PAR_FORMATEUR = 20000, 50, 40, 3
PRIX: Final = (20, 18, 16, 14)
COUT_DE_PRODUCTION, SALAIRE, SALAIRE_D_ELEVE = 5, 600, 300
VARIABLES: Final = Variables(
    (
        *numerotees("T", SEMAINES),
        *numerotees("F", SEMAINES),
        *numerotees("I", SEMAINES - 1),
        *numerotees("E", SEMAINES - 1),
    )
)


@dataclass(frozen=True, slots=True)
class Plan:
    """Le profit et, semaine par semaine, qui fait quoi."""

    profit: sp.Expr
    travailleurs: tuple[sp.Expr, ...]
    formes: tuple[sp.Expr, ...]
    instructeurs: tuple[sp.Expr, ...]
    eleves: tuple[sp.Expr, ...]


def marge_d_un_travailleur(semaine: int) -> int:
    """Ce que rapporte une semaine de production : 50 radios, cout deduit."""
    return CADENCE * (PRIX[semaine - 1] - COUT_DE_PRODUCTION)


def _objectif() -> dict[str, int]:
    objectif = {f"T{i}": marge_d_un_travailleur(i) for i in range(1, SEMAINES + 1)}
    objectif |= {f"F{i}": -SALAIRE for i in range(1, SEMAINES + 1)}
    return objectif | {f"E{i}": -SALAIRE_D_ELEVE for i in range(1, SEMAINES)}


@cache
def programme() -> ProgrammeGeneral:
    """Le modele de la planche 22."""
    ligne = VARIABLES.ligne
    production = dict.fromkeys(numerotees("T", SEMAINES), CADENCE)
    contraintes = [
        Contrainte(ligne(production), Sens.EGAL, A_LIVRER),
        Contrainte(ligne({"F1": 1}), Sens.EGAL, PERMANENTS),
        Contrainte(ligne({f"F{SEMAINES}": 1, f"T{SEMAINES}": -1}), Sens.SUPERIEUR, 0),
    ]
    for i in range(1, SEMAINES):
        effectif = {f"F{i + 1}": 1, f"F{i}": -1, f"E{i}": -1}
        statut = {f"F{i}": 1, f"T{i}": -1, f"I{i}": -1}
        formation = {f"E{i}": 1, f"I{i}": -ELEVES_PAR_FORMATEUR}
        contraintes.append(Contrainte(ligne(effectif), Sens.EGAL, 0))
        contraintes.append(Contrainte(ligne(statut), Sens.SUPERIEUR, 0))
        contraintes.append(Contrainte(ligne(formation), Sens.INFERIEUR, 0))
    return ProgrammeGeneral(Optimisation.MAX, ligne(_objectif()), tuple(contraintes))


@cache
def plan() -> Plan:
    """Un plan de profit maximal, par le simplexe."""
    profit, point = optimum_fini(programme())
    lire = VARIABLES.lire
    return Plan(
        profit, lire(point, "T"), lire(point, "F"), lire(point, "I"), lire(point, "E")
    )


def radios(le_plan: Plan) -> sp.Expr:
    """Le nombre de radios que le plan produit."""
    return CADENCE * sum(le_plan.travailleurs)


def recettes_nettes(le_plan: Plan) -> sp.Expr:
    """Les ventes, cout de production deduit, semaine par semaine."""
    marges = (marge_d_un_travailleur(i) for i in range(1, SEMAINES + 1))
    return sum(m * t for m, t in zip(marges, le_plan.travailleurs, strict=True))


def salaires(le_plan: Plan) -> sp.Expr:
    """Les salaires des formes et des eleves, sur les quatre semaines."""
    return SALAIRE * sum(le_plan.formes) + SALAIRE_D_ELEVE * sum(le_plan.eleves)


def profit_en_entiers() -> sp.Expr | None:
    """Le profit maximal quand toutes les variables sont entieres."""
    entieres = frozenset(range(len(VARIABLES.noms)))
    return resoudre_en_variables_entieres(ProgrammeEntier(programme(), entieres)).valeur
