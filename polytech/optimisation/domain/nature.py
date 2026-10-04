"""Chapitre I.4.1 et I.4.2 : ce qu'est l'ensemble `D`, et ce qu'est l'optimum.

**L'ensemble `D` des solutions admissibles** est un polytope, un polyedre
convexe non vide mais non borne, ou l'ensemble vide. Rien d'autre.

**La solution optimale**, selon le cas :

* `D` polytope : elle est unique, en un sommet ; ou bien il y en a une
  infinite, les points d'une face ;
* `D` non borne : en plus, il se peut qu'aucune solution optimale ne
  soit a distance finie -- `z` decroit sans fin le long d'un rayon ;
* `D` vide : aucune solution.

Aucun solveur n'est appele. Un domaine non vide en forme standard a un
sommet, donc une solution de base admissible. Un domaine est non borne
s'il contient un rayon `x + lambda d` : `T d = 0`, `d >= 0`, `d != 0`.
Normalises par `somme(d) = 1`, ces rayons forment un polytope dont les
sommets -- les **rayons extremes** -- sont eux-memes des solutions de
base admissibles, du systeme augmente d'une ligne.
"""

from __future__ import annotations

from enum import Enum

import sympy as sp

from polytech.optimisation.domain.bases import solutions_de_base
from polytech.optimisation.domain.formes import FormeStandard


class NatureDuDomaine(Enum):
    """Les trois situations possibles pour `D` (propriete I.4.1)."""

    POLYTOPE = "polytope"
    NON_BORNE = "polyedre convexe non vide et non borne"
    VIDE = "ensemble vide"


class NatureDeLOptimum(Enum):
    """Les situations possibles pour la solution optimale (propriete I.4.2)."""

    UNIQUE = "solution optimale unique, en un sommet"
    INFINITE = "infinite de solutions optimales, sur une face"
    NON_BORNEE = "pas de solution optimale finie"
    SANS_SOLUTION = "D est vide"


def rayons_extremes(forme: FormeStandard) -> tuple[sp.Matrix, ...]:
    """Les directions extremes de `D`, normalisees par `somme(d) = 1`."""
    m, n = forme.matrice.shape
    augmentee = forme.matrice.col_join(sp.ones(1, n))
    if augmentee.rank() == m:
        return ()
    systeme = FormeStandard(
        couts=forme.couts,
        matrice=augmentee,
        seconds_membres=sp.zeros(m, 1).col_join(sp.ones(1, 1)),
    )
    rayons: list[sp.Matrix] = []
    for solution in solutions_de_base(systeme):
        if solution.admissible and solution.point not in rayons:
            rayons.append(solution.point)
    return tuple(rayons)


def _sommets(forme: FormeStandard) -> list[sp.Matrix]:
    points: list[sp.Matrix] = []
    for solution in solutions_de_base(forme):
        if solution.admissible and solution.point not in points:
            points.append(solution.point)
    return points


def nature_du_domaine(forme: FormeStandard) -> NatureDuDomaine:
    """Polytope, non borne, ou vide."""
    if not _sommets(forme):
        return NatureDuDomaine.VIDE
    if rayons_extremes(forme):
        return NatureDuDomaine.NON_BORNE
    return NatureDuDomaine.POLYTOPE


def nature_de_l_optimum(forme: FormeStandard) -> NatureDeLOptimum:
    """Unique, infinite sur une face, non bornee, ou sans solution."""
    sommets = _sommets(forme)
    if not sommets:
        return NatureDeLOptimum.SANS_SOLUTION
    pentes = [(forme.couts * d)[0, 0] for d in rayons_extremes(forme)]
    if any(pente < 0 for pente in pentes):
        return NatureDeLOptimum.NON_BORNEE
    valeurs = [(forme.couts * x)[0, 0] for x in sommets]
    optimaux = valeurs.count(min(valeurs))
    if optimaux > 1 or any(pente == 0 for pente in pentes):
        return NatureDeLOptimum.INFINITE
    return NatureDeLOptimum.UNIQUE
