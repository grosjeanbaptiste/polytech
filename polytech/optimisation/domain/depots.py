"""Chapitre XIV.4.4 : le probleme d'implantation de depots (« Facility Location »).

Quels depots ouvrir parmi des sites potentiels, et comment leur affecter
les clients ? `x_j = 1` si le depot `j` est ouvert, de cout fixe `f_j` ;
`g_ij` est le gain d'approvisionner le client `i` par le depot `j`.

a) **Sans capacite** (UFLP) : `y_ij = 1` si le client `i` est servi par
   le depot `j`.

       max sum g_ij y_ij - sum f_j x_j
       (1) sum_j y_ij = 1     chaque client rattache a un seul depot
       (2) y_ij <= x_j        et seulement a un depot ouvert

b) **Avec capacites** (CFLP) : le client `i` demande `d_i`, le depot `j`
   peut fournir `q_j` ; `y_ij >= 0` est la quantite livree, `g_ij` un gain
   unitaire. C'est un probleme **mixte**.

       (1) sum_j y_ij = d_i   (2) sum_i y_ij <= q_j x_j

Les variables sont rangees `x_1 ... x_m`, puis les `y_ij` client par client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.combinatoire import bornes_binaires
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence

type Gains = Sequence[Sequence[int]]


def _objectif(gains: Gains, couts_fixes: Sequence[int]) -> tuple[int, ...]:
    depots = len(couts_fixes)
    if any(len(ligne) != depots for ligne in gains):
        msg = f"gains pour {len(gains[0])} depots, couts fixes pour {depots}"
        raise ValueError(msg)
    return (*(-f for f in couts_fixes), *(g for ligne in gains for g in ligne))


def _ligne(variables: int, termes: dict[int, int]) -> tuple[int, ...]:
    return tuple(termes.get(rang, 0) for rang in range(variables))


def _livraisons(clients: int, depots: int, demandes: Sequence[int]) -> list[Contrainte]:
    """(1) : `sum_j y_ij` egale la demande du client `i`."""
    n = depots + clients * depots
    return [
        Contrainte(
            _ligne(n, {depots + i * depots + j: 1 for j in range(depots)}),
            Sens.EGAL,
            demandes[i],
        )
        for i in range(clients)
    ]


def implantation_sans_capacite(
    gains: Gains, couts_fixes: Sequence[int]
) -> ProgrammeEntier:
    """(UFLP) : toutes les variables binaires."""
    objectif = _objectif(gains, couts_fixes)
    clients, depots, n = len(gains), len(couts_fixes), len(objectif)
    ouverts = [
        Contrainte(_ligne(n, {j: -1, depots + i * depots + j: 1}), Sens.INFERIEUR, 0)
        for i in range(clients)
        for j in range(depots)
    ]
    contraintes = (
        *_livraisons(clients, depots, (1,) * clients),
        *ouverts,
        *bornes_binaires(n),
    )
    programme = ProgrammeGeneral(Optimisation.MAX, objectif, contraintes)
    return ProgrammeEntier(programme, frozenset(range(n)))


def implantation_avec_capacites(
    gains: Gains,
    couts_fixes: Sequence[int],
    demandes: Sequence[int],
    capacites: Sequence[int],
) -> ProgrammeEntier:
    """(CFLP) : les `x_j` binaires, les quantites `y_ij` continues."""
    objectif = _objectif(gains, couts_fixes)
    clients, depots, n = len(gains), len(couts_fixes), len(objectif)
    limites = [
        Contrainte(
            _ligne(
                n,
                {j: -capacites[j]}
                | {depots + i * depots + j: 1 for i in range(clients)},
            ),
            Sens.INFERIEUR,
            0,
        )
        for j in range(depots)
    ]
    binaires = bornes_binaires(n)[:depots]
    contraintes = (*_livraisons(clients, depots, demandes), *limites, *binaires)
    programme = ProgrammeGeneral(Optimisation.MAX, objectif, contraintes)
    return ProgrammeEntier(programme, frozenset(range(depots)))
