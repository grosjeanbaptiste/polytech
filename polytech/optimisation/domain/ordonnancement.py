"""Chapitre XIV.4.5 : deux problemes de production en variables mixtes.

**Machines paralleles.** `n` taches de duree `p_i`, `m` machines
identiques ; `x_ik = 1` si la tache `i` va sur la machine `k`. Le
`min max` des charges se linearise par une variable continue `y` :

    min y,   sum_i p_i x_ik <= y   pour chaque machine k.

Le cours n'ecrit pas que chaque tache va sur une machine et une seule,
`sum_k x_ik = 1` : la definition des `x_ik` le sous-entend, et sans cette
contrainte `y = 0` serait optimal. Elle est ecrite ici.

**« Flow shop » de permutation.** `m` machines en serie, le meme ordre
sur toutes. `x_ij = 1` si la tache `i` est en position `j` ; `y_jk`,
l'attente de la tache en position `j` entre les machines `k` et `k + 1` ;
`z_jk`, l'inactivite de la machine `k` avant la tache en position `j`.
La relation de la figure XIV.12 lie deux positions successives, et l'on
minimise `sum_j z_jm` -- la duree totale a la constante `sum_i p_im` pres.

**Un ecart au cours, etabli.** La formulation de la page 319 ne fixe ni
`y_1k` ni `z_1k`. Or la premiere tache n'attend jamais (`y_1k = 0`) et la
machine `k + 1` reste inactive tant que cette tache n'a pas traverse les
precedentes (`z_1,k+1 = sum(p[1]k', k' <= k)`). Sans ces conditions
initiales, cette inactivite est gratuite et la duree sous-estimee : un
test le montre. Elles sont ajoutees ici.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.combinatoire import bornes_binaires
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.flow_shop import contraintes_du_flow_shop
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence

type Durees = Sequence[Sequence[int]]


def machines_paralleles(durees: Sequence[int], machines: int) -> ProgrammeEntier:
    """`min y` : la charge de la machine la plus chargee."""
    taches = len(durees)
    binaires = taches * machines
    n = binaires + 1
    affectations = [
        Contrainte(
            (*(int(rang // machines == i) for rang in range(binaires)), 0),
            Sens.EGAL,
            1,
        )
        for i in range(taches)
    ]
    charges = [
        Contrainte(
            (
                *(
                    durees[rang // machines] * int(rang % machines == k)
                    for rang in range(binaires)
                ),
                -1,
            ),
            Sens.INFERIEUR,
            0,
        )
        for k in range(machines)
    ]
    contraintes = (*affectations, *charges, *bornes_binaires(n)[:binaires])
    objectif = (0,) * binaires + (1,)
    programme = ProgrammeGeneral(Optimisation.MIN, objectif, contraintes)
    return ProgrammeEntier(programme, frozenset(range(binaires)))


def duree_totale(durees: Durees, ordre: Sequence[int]) -> int:
    """La duree d'un ordre donne (taches comptees de 1), par simulation."""
    machines = len(durees[0])
    fins = [0] * machines
    for tache in ordre:
        libre = 0
        for k in range(machines):
            libre = max(libre, fins[k]) + durees[tache - 1][k]
            fins[k] = libre
    return fins[-1]


def flow_shop_de_permutation(
    durees: Durees,
    *,
    conditions_initiales: bool = True,
) -> tuple[ProgrammeEntier, int]:
    """Le programme mixte de la page 319, et la constante `sum_i p_im`."""
    taches, machines = len(durees), len(durees[0])
    binaires = taches * taches
    n = binaires + taches * (machines - 1) * 2
    contraintes = (
        *contraintes_du_flow_shop(durees, conditions_initiales=conditions_initiales),
        *bornes_binaires(n)[:binaires],
    )
    derniere = n - taches * (machines - 1) + machines - 2
    inactivites = range(derniere, n, machines - 1)
    objectif = tuple(int(rang in inactivites) for rang in range(n))
    programme = ProgrammeGeneral(Optimisation.MIN, objectif, contraintes)
    constante = sum(ligne[-1] for ligne in durees)
    return ProgrammeEntier(programme, frozenset(range(binaires))), constante
