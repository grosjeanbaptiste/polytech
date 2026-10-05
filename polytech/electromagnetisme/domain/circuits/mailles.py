"""Les equations de Kirchhoff et la methode des courants de maille.

Le syllabus d'electricite generale (13.16 a 13.18). A chaque noeud, la
somme algebrique des courants est nulle (13.17.1) ; le long d'une maille,
la somme des forces electromotrices rencontrees egale celle des chutes de
tension, `somme E_kj = somme R_kj I_kj` (13.17.3).

Les courants de maille (courants cycliques de Maxwell) sont ceux des
maillons du co-arbre, chacun parcourant sa maille de bout en bout. Ils
satisfont d'office les equations de premiere espece : le courant d'une
branche de l'arbre est la somme algebrique des courants de maille qui la
parcourent, et il ne reste que `b - (n - 1)` inconnues.

Une branche va de `de` vers `vers` ; son courant et sa force
electromotrice sont positifs dans ce sens.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.electromagnetisme.domain.circuits.arbre import maille_fondamentale

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from polytech.electromagnetisme.domain.circuits.arbre import Pas


@dataclass(frozen=True, slots=True)
class Branche:
    """Une branche : ses deux noeuds, sa resistance, sa force electromotrice."""

    de: str
    vers: str
    resistance: Any
    fem: Any = 0


def boucle_par_les_noeuds(
    branches: Sequence[Branche], parcours: Sequence[str]
) -> tuple[Pas, ...]:
    """Les pas `(rang, sens)` d'un parcours donne par ses noeuds successifs."""
    pas = []
    for ici, la in pairwise(parcours):
        suivies = [
            (rang, 1 if (b.de, b.vers) == (ici, la) else -1)
            for rang, b in enumerate(branches)
            if {b.de, b.vers} == {ici, la}
        ]
        if not suivies:
            msg = f"aucune branche entre {ici} et {la}"
            raise ValueError(msg)
        pas.append(suivies[0])
    return tuple(pas)


def residu_d_une_maille(
    branches: Sequence[Branche], boucle: Sequence[Pas], courants: Sequence[Any]
) -> sp.Expr:
    """`somme E_kj - somme R_kj I_kj` le long de la maille : nul par (13.17.3)."""
    termes = (
        sens * (branches[rang].fem - branches[rang].resistance * courants[rang])
        for rang, sens in boucle
    )
    return sp.expand(sp.Add(*termes))


def residus_des_noeuds(
    branches: Sequence[Branche], courants: Sequence[Any]
) -> dict[str, sp.Expr]:
    """La somme des courants qui partent de chaque noeud : nulle par (13.17.1)."""
    sortant: dict[str, sp.Expr] = {}
    for branche, courant in zip(branches, courants, strict=True):
        sortant[branche.de] = sortant.get(branche.de, 0) + courant
        sortant[branche.vers] = sortant.get(branche.vers, 0) - courant
    return {noeud: sp.simplify(somme) for noeud, somme in sortant.items()}


def _mailles(
    branches: Sequence[Branche], arbre: Collection[int]
) -> list[tuple[Pas, ...]]:
    graphe = [(b.de, b.vers) for b in branches]
    maillons = [rang for rang in range(len(branches)) if rang not in arbre]
    return [maille_fondamentale(graphe, arbre, maillon) for maillon in maillons]


def courants_des_branches(
    branches: Sequence[Branche], arbre: Collection[int], de_maille: Sequence[Any]
) -> tuple[sp.Expr, ...]:
    """Le courant de chaque branche, somme des courants de maille qui y passent."""
    total = [sp.Integer(0)] * len(branches)
    for maille, courant in zip(_mailles(branches, arbre), de_maille, strict=True):
        for rang, sens in maille:
            total[rang] += sens * courant
    return tuple(total)


def resoudre_par_les_mailles(
    branches: Sequence[Branche], arbre: Collection[int]
) -> tuple[sp.Expr, ...]:
    """Les courants de branche, par `b - (n - 1)` equations de deuxieme espece."""
    mailles = _mailles(branches, arbre)
    inconnues = sp.symbols(f"J_0:{len(mailles)}", real=True)
    courants = courants_des_branches(branches, arbre, inconnues)
    equations = [residu_d_une_maille(branches, m, courants) for m in mailles]
    solutions = sp.linsolve(equations, inconnues)
    if not solutions or next(iter(solutions)).free_symbols & set(inconnues):
        msg = "le reseau n'a pas de solution unique : une maille sans resistance"
        raise ValueError(msg)
    valeurs = dict(zip(inconnues, next(iter(solutions)), strict=True))
    return tuple(sp.simplify(courant.subs(valeurs)) for courant in courants)
