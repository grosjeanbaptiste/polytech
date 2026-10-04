"""Chapitre XIV.4.2 : le voyageur de commerce, et l'affectation qui le relaxe.

`n` villes, `c_ij` le cout du trajet `i -> j`, `x_ij = 1` si le tour le
contient. Le probleme (TSP) :

    min sum c_ij x_ij
    (1) sum_j x_ij = 1   pour tout i        (on quitte chaque ville une fois)
    (2) sum_i x_ij = 1   pour tout j        (on y arrive une fois)
    (3) aucun sous-tour.

Sans (3), c'est le probleme d'**affectation** (AP) : une permutation, qui
peut se decomposer en plusieurs circuits. (AP) est une relaxation de
(TSP) -- de classe P, la ou (TSP) est NP-difficile -- et sa valeur sert
de fonction d'evaluation dans une procedure par separation et evaluation.

Deux ecritures de (3), toutes deux du cours : par les **coupes**,
`sum(x_ij, i dans Q, j hors de Q) >= 1` ; par les **cliques**,
`sum(x_ij, i et j dans Q) <= |Q| - 1`, pour `2 <= |Q| <= n - 2`.
"""

from __future__ import annotations

from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING

from polytech.optimisation.domain.combinatoire import programme_binaire
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.entiers import ProgrammeEntier

type Couts = Sequence[Sequence[int]]


class SousTours(Enum):
    """Les deux ecritures de l'interdiction des sous-tours (page 314)."""

    COUPES = "sum(x_ij, i dans Q, j hors de Q) >= 1"
    CLIQUES = "sum(x_ij, i et j dans Q) <= |Q| - 1"


def arcs(villes: int) -> tuple[tuple[int, int], ...]:
    """Les trajets `i -> j`, `i != j`, dans l'ordre des variables `x_ij`."""
    numeros = range(1, villes + 1)
    return tuple((i, j) for i in numeros for j in numeros if i != j)


def sous_tours(villes: int) -> tuple[frozenset[int], ...]:
    """Les sous-ensembles `Q` avec `2 <= |Q| <= n - 2`."""
    numeros = range(1, villes + 1)
    return tuple(
        frozenset(q)
        for taille in range(2, villes - 1)
        for q in combinations(numeros, taille)
    )


def _degres(villes: int) -> list[Contrainte]:
    trajets = arcs(villes)
    departs = [
        Contrainte(tuple(int(i == v) for i, _ in trajets), Sens.EGAL, 1)
        for v in range(1, villes + 1)
    ]
    arrivees = [
        Contrainte(tuple(int(j == v) for _, j in trajets), Sens.EGAL, 1)
        for v in range(1, villes + 1)
    ]
    return departs + arrivees


def _objectif(couts: Couts) -> tuple[int, ...]:
    if any(len(ligne) != len(couts) for ligne in couts):
        msg = f"matrice des couts non carree : {len(couts)} lignes"
        raise ValueError(msg)
    return tuple(couts[i - 1][j - 1] for i, j in arcs(len(couts)))


def affectation(couts: Couts) -> ProgrammeEntier:
    """(AP) : les contraintes (1) et (2) seules."""
    return programme_binaire(Optimisation.MIN, _objectif(couts), _degres(len(couts)))


def _sans_sous_tour(villes: int, q: frozenset[int], ecriture: SousTours) -> Contrainte:
    trajets = arcs(villes)
    if ecriture is SousTours.COUPES:
        sortants = tuple(int(i in q and j not in q) for i, j in trajets)
        return Contrainte(sortants, Sens.SUPERIEUR, 1)
    internes = tuple(int(i in q and j in q) for i, j in trajets)
    return Contrainte(internes, Sens.INFERIEUR, len(q) - 1)


def voyageur_de_commerce(
    couts: Couts,
    ecriture: SousTours = SousTours.CLIQUES,
) -> ProgrammeEntier:
    """(TSP) : l'affectation, plus une contrainte (3) par sous-ensemble `Q`."""
    villes = len(couts)
    interdits = [_sans_sous_tour(villes, q, ecriture) for q in sous_tours(villes)]
    contraintes = [*_degres(villes), *interdits]
    return programme_binaire(Optimisation.MIN, _objectif(couts), contraintes)


def sequencement_avec_reglages(
    durees: Sequence[int],
    reglages: Couts,
) -> tuple[ProgrammeEntier, int]:
    """XIV.4.5 (1) : l'ordre des taches sur une machine, temps de reglage minimaux.

    Une tache fictive 0, sans reglage ni avant ni apres, ferme le circuit :
    c'est un (TSP) sur `n + 1` villes. La somme des durees est constante ;
    elle est rendue a part.
    """
    fictive = (0,) * (len(durees) + 1)
    couts = (fictive, *((0, *ligne) for ligne in reglages))
    return voyageur_de_commerce(couts), sum(durees)
