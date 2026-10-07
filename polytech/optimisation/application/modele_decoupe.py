"""Modelisation, probleme II : decouper des rouleaux de papier.

Des rouleaux de 2,50 m doivent fournir quatre largeurs en quantites
commandees. Une variable par facon de couper un rouleau : la planche en
dresse vingt-cinq. On minimise soit la perte, soit le nombre de
decoupes.
"""

from __future__ import annotations

import itertools
from functools import cache
from typing import Final

import sympy as sp

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

Motif = tuple[int, ...]
ROULEAU: Final = sp.Rational(5, 2)
LARGEURS: Final = (
    sp.Rational(6, 5),
    sp.Integer(1),
    sp.Rational(22, 25),
    sp.Rational(1, 2),
)
COMMANDES: Final = (10, 15, 30, 8)
"""Planche 6 : 1,20 m, 1,00 m, 0,88 m et 0,50 m."""
MOTIFS: Final[tuple[Motif, ...]] = (
    (2, 0, 0, 0),
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (1, 0, 0, 2),
    (0, 2, 0, 0),
    (0, 2, 0, 1),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 1, 1, 1),
    (0, 1, 0, 1),
    (0, 1, 0, 2),
    (0, 1, 0, 3),
    (0, 0, 2, 0),
    (0, 0, 2, 1),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 1, 2),
    (0, 0, 1, 3),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 0, 0, 3),
    (0, 0, 0, 4),
    (0, 0, 0, 5),
)
"""Planches 7 et 8, dans leur ordre : `x_1` a `x_25`."""
SOLUTIONS: Final = ({4: 10, 8: 7, 10: 1, 15: 9, 18: 1}, {1: 5, 8: 7, 15: 15, 12: 1})
"""Planche 11 : le nombre de rouleaux coupes selon chaque motif."""


def largeur(motif: Motif) -> sp.Expr:
    """La largeur que les pieces du motif occupent."""
    return sum(
        (n * cote for n, cote in zip(motif, LARGEURS, strict=True)), sp.Integer(0)
    )


def toutes_les_decoupes() -> tuple[Motif, ...]:
    """Toutes les facons de tirer au moins une piece d'un rouleau."""
    plafonds = [int(ROULEAU / cote) for cote in LARGEURS]
    candidats = itertools.product(*(range(p + 1) for p in plafonds))
    return tuple(m for m in candidats if any(m) and largeur(m) <= ROULEAU)


def perte(motif: Motif) -> sp.Expr:
    """La chute : ce qui reste du rouleau."""
    return ROULEAU - largeur(motif)


def decoupes(motif: Motif) -> int:
    """Un trait par piece, sauf le dernier si la chute est nulle."""
    return sum(motif) - (1 if perte(motif) == 0 else 0)


def _objectifs() -> dict[str, tuple[sp.Expr | int, ...]]:
    return {
        "perte": tuple(perte(m) for m in MOTIFS),
        "decoupes": tuple(decoupes(m) for m in MOTIFS),
        "rouleaux": (1,) * len(MOTIFS),
    }


@cache
def relaxation(objectif: str) -> sp.Expr:
    """Le minimum de cet objectif, les nombres de rouleaux n'etant pas entiers."""
    if objectif not in _objectifs():
        msg = f"objectif inconnu : {objectif}"
        raise KeyError(msg)
    commandes = tuple(
        Contrainte(tuple(m[k] for m in MOTIFS), Sens.EGAL, COMMANDES[k])
        for k in range(len(LARGEURS))
    )
    modele = ProgrammeGeneral(Optimisation.MIN, _objectifs()[objectif], commandes)
    return optimum_fini(modele)[0]
