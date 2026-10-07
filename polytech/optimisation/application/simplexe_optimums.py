"""Exercice II du jeu « algorithme simplexe » : toutes les solutions optimales.

`min x + y + z` sous `x + y >= 1` et `x + y + z >= 2`. L'optimum vaut 2 et
il est atteint en plusieurs sommets : toute combinaison convexe de ces
sommets est optimale.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence

Point = tuple[sp.Expr, ...]
PROGRAMME: Final = ProgrammeGeneral(
    Optimisation.MIN,
    (1, 1, 1),
    (
        Contrainte((1, 1, 0), Sens.SUPERIEUR, 1),
        Contrainte((1, 1, 1), Sens.SUPERIEUR, 2),
    ),
)
VARIABLES: Final = 3


def valeur(point: Sequence[object]) -> sp.Expr:
    """`x + y + z` en ce point."""
    return PROGRAMME.valeur(point)


def valeur_optimale() -> sp.Expr:
    """Le minimum, par la bibliotheque."""
    return optimum_fini(PROGRAMME)[0]


def est_admissible(point: Sequence[object]) -> bool:
    """Vrai si le point est positif et respecte les deux contraintes."""
    positif = all(sp.sympify(v) >= 0 for v in point)
    return positif and all(c.ecart(point) >= 0 for c in PROGRAMME.contraintes)


def sommets_du_domaine() -> tuple[Point, ...]:
    """Les sommets : une solution de base admissible par base reguliere."""
    standard = forme_standard(PROGRAMME)
    equations, colonnes = standard.matrice.shape
    sommets: list[Point] = []
    for base in itertools.combinations(range(colonnes), equations):
        carree = standard.matrice[:, list(base)]
        if carree.det() == 0:
            continue
        solution = carree.inv() * standard.seconds_membres
        if any(v < 0 for v in solution):
            continue
        complet = [sp.Integer(0)] * colonnes
        for rang, indice in enumerate(base):
            complet[indice] = solution[rang]
        point = tuple(complet[:VARIABLES])
        if point not in sommets:
            sommets.append(point)
    return tuple(sommets)


def sommets_optimaux() -> tuple[Point, ...]:
    """Ceux des sommets dont la valeur est le minimum."""
    minimum = valeur_optimale()
    return tuple(s for s in sommets_du_domaine() if valeur(s) == minimum)


def combinaison(poids: Sequence[sp.Expr]) -> Point:
    """La combinaison convexe des sommets optimaux, avec ces poids."""
    sommets = sorted(sommets_optimaux(), key=sp.default_sort_key)
    if sum(poids) != 1 or any(p < 0 for p in poids) or len(poids) != len(sommets):
        msg = f"poids non convexes pour {len(sommets)} sommets : {tuple(poids)}"
        raise ValueError(msg)
    return tuple(
        sum((p * s[k] for p, s in zip(poids, sommets, strict=True)), sp.Integer(0))
        for k in range(VARIABLES)
    )
