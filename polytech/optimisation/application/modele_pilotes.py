"""Modelisation, probleme IV : former des pilotes ou les faire voler.

En `n` semaines il faut 700 semaines de vol. Chaque semaine, un pilote
vole (`V`), se repose (`R`) ou instruit (`I`) ; un instructeur forme
vingt pilotes, disponibles la semaine suivante. Qui a vole se repose la
semaine d'apres, et le nombre de pilotes en vol ne diminue jamais.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.application.modele_variables import Variables, numerotees
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence

PILOTES, A_VOLER, FORMES = 320, 700, 20
COUT_VOL, COUT_REPOS, COUT_INSTRUCTION = 80, 5, 200


@dataclass(frozen=True, slots=True)
class Plan:
    """Semaine par semaine : ceux qui volent, se reposent, instruisent."""

    vols: tuple[int, ...]
    repos: tuple[int, ...]
    instructeurs: tuple[int, ...]


PLAN_A_QUATRE: Final = Plan((160, 180, 180, 180), (158, 180, 180, 180), (2, 0, 0, 0))
"""Deux instructeurs la premiere semaine portent l'effectif a 360."""


def variables(semaines: int) -> Variables:
    """`V`, `R` et `I`, une de chaque par semaine."""
    return Variables(tuple(n for p in "VRI" for n in numerotees(p, semaines)))


def _regles(semaines: int) -> list[Contrainte]:
    """Tout sauf l'objectif de 700 : effectifs, repos, vols non decroissants."""
    ligne = variables(semaines).ligne
    regles = [Contrainte(ligne({"V1": 1, "R1": 1, "I1": 1}), Sens.EGAL, PILOTES)]
    for i in range(1, semaines):
        suite = {f"V{i + 1}": 1, f"R{i + 1}": 1, f"I{i + 1}": 1}
        avant = {f"V{i}": -1, f"R{i}": -1, f"I{i}": -(FORMES + 1)}
        regles.append(Contrainte(ligne(suite | avant), Sens.EGAL, 0))
        repos = ligne({f"R{i + 1}": 1, f"V{i}": -1})
        vols = ligne({f"V{i + 1}": 1, f"V{i}": -1})
        regles.append(Contrainte(repos, Sens.SUPERIEUR, 0))
        regles.append(Contrainte(vols, Sens.SUPERIEUR, 0))
    return regles


def _tous_les_vols(semaines: int) -> tuple[sp.Expr | int, ...]:
    return variables(semaines).ligne(dict.fromkeys(numerotees("V", semaines), 1))


def programme(semaines: int) -> ProgrammeGeneral:
    """Le modele de la planche 20."""
    couts = {"V": COUT_VOL, "R": COUT_REPOS, "I": COUT_INSTRUCTION}
    noms = variables(semaines).noms
    objectif = tuple(couts[nom[0]] for nom in noms)
    total = Contrainte(_tous_les_vols(semaines), Sens.EGAL, A_VOLER)
    return ProgrammeGeneral(Optimisation.MIN, objectif, (total, *_regles(semaines)))


@cache
def relaxation(semaines: int) -> sp.Expr:
    """Le cout minimal quand on oublie que les pilotes sont entiers."""
    return optimum_fini(programme(semaines))[0]


def cout(instructeurs: Sequence[sp.Expr | int]) -> sp.Expr:
    """Le cout d'un plan ne depend que de ses instructeurs.

    L'effectif grandit de 20 par instructeur ; tout pilote qui ne vole ni
    n'instruit se repose. D'ou les repos, puis le cout.
    """
    effectifs = list(
        itertools.accumulate(
            instructeurs[:-1], lambda e, i: e + FORMES * i, initial=PILOTES
        )
    )
    repos = sum(effectifs) - A_VOLER - sum(instructeurs)
    total = (
        COUT_VOL * A_VOLER + COUT_REPOS * repos + COUT_INSTRUCTION * sum(instructeurs)
    )
    return sp.sympify(total)


def _point(plan: Plan) -> tuple[int, ...]:
    return (*plan.vols, *plan.repos, *plan.instructeurs)


def est_admissible(plan: Plan) -> bool:
    """Vrai si le plan respecte toutes les contraintes du modele."""
    contraintes = programme(len(plan.vols)).contraintes
    return all(c.est_satisfaite(_point(plan)) for c in contraintes)


def cout_du_plan(plan: Plan) -> sp.Expr:
    """L'objectif du modele, evalue sur ce plan."""
    semaines = len(plan.vols)
    return programme(semaines).valeur(_point(plan))


@cache
def vols_possibles(semaines: int, instructeurs: tuple[int, ...]) -> sp.Expr:
    """Le plus de semaines de vol qu'on puisse fournir, les instructeurs etant fixes."""
    ligne = variables(semaines).ligne
    fixes = [
        Contrainte(ligne({f"I{i}": 1}), Sens.EGAL, nombre)
        for i, nombre in enumerate(instructeurs, start=1)
    ]
    contraintes = (*_regles(semaines), *fixes)
    modele = ProgrammeGeneral(Optimisation.MAX, _tous_les_vols(semaines), contraintes)
    return optimum_fini(modele)[0]


def choix_moins_chers(semaines: int, plafond: int) -> tuple[tuple[int, ...], ...]:
    """Tous les choix entiers d'instructeurs qui coutent moins que ce plafond."""
    base = cout((0,) * semaines)
    dernier = cout((*(0,) * (semaines - 1), 1)) - base
    maximum = int((plafond - base) / dernier)
    choix = itertools.product(range(maximum + 1), repeat=semaines)
    return tuple(c for c in choix if cout(c) < plafond)


def plan_decale(plan: Plan, semaine: int, vols: int) -> Plan:
    """Le meme plan, un nombre de vols change : pour eprouver les contraintes."""
    modifies = list(plan.vols)
    modifies[semaine - 1] = vols
    return replace(plan, vols=tuple(modifies))
