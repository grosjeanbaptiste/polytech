"""Chapitre XIV.4.1 et XIV.4.3 : sac a dos, couverture, partition, emballage.

Les variables binaires « 0 - 1 » disent si un evenement se realise : un
objet est pris, un sous-ensemble est retenu. Quatre problemes types :

* le **sac a dos** (KP) : `max sum p_j x_j`, `sum w_j x_j <= W` -- une
  seule contrainte ; chargement d'un bateau, selection d'investissements ;
* la **couverture** (SCP) : choisir des sous-ensembles `P_j` de cout
  minimal tels que chaque objet apparaisse **au moins** une fois ;
* la **partition** (SPP) : chaque objet **exactement** une fois ;
* l'**emballage** : des sous-ensembles disjoints de valeur maximale,
  chaque objet **au plus** une fois.

Une variable binaire est une variable entiere bornee par 1 : la borne
s'ecrit en contrainte, et la relaxation lineaire la garde.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence


def bornes_binaires(variables: int) -> tuple[Contrainte, ...]:
    """`x_j <= 1` pour chaque variable : avec l'integrite, `x_j = 0` ou `1`."""
    return tuple(
        Contrainte(tuple(int(i == j) for i in range(variables)), Sens.INFERIEUR, 1)
        for j in range(variables)
    )


def programme_binaire(
    optimisation: Optimisation,
    objectif: Sequence[object],
    contraintes: Sequence[Contrainte],
) -> ProgrammeEntier:
    """Un programme dont toutes les variables sont binaires."""
    n = len(objectif)
    toutes = (*contraintes, *bornes_binaires(n))
    programme = ProgrammeGeneral(optimisation, tuple(objectif), toutes)
    return ProgrammeEntier(programme, frozenset(range(n)))


def sac_a_dos(
    valeurs: Sequence[int],
    poids: Sequence[int],
    capacite: int,
) -> ProgrammeEntier:
    """(KP) : les objets de plus grande valeur totale, sans depasser la capacite."""
    if len(poids) != len(valeurs):
        msg = f"{len(poids)} poids pour {len(valeurs)} valeurs"
        raise ValueError(msg)
    charge = Contrainte(tuple(poids), Sens.INFERIEUR, capacite)
    return programme_binaire(Optimisation.MAX, valeurs, (charge,))


def sac_a_dos_est_non_trivial(poids: Sequence[int], capacite: int) -> bool:
    """Chaque objet entre seul, et tous ensemble n'entrent pas (page 314)."""
    return all(w <= capacite for w in poids) and sum(poids) > capacite


def _incidences(
    objets: int,
    sous_ensembles: Sequence[frozenset[int]],
    sens: Sens,
) -> tuple[Contrainte, ...]:
    """Une contrainte par objet : `sum t_ij x_j  (>=, =, <=)  1`."""
    hors = sorted({i for p in sous_ensembles for i in p if not 1 <= i <= objets})
    if hors:
        msg = f"objet hors de l'ensemble : {hors} pour {objets} objets"
        raise ValueError(msg)
    return tuple(
        Contrainte(tuple(int(i in p) for p in sous_ensembles), sens, 1)
        for i in range(1, objets + 1)
    )


def couverture(
    objets: int,
    sous_ensembles: Sequence[frozenset[int]],
    couts: Sequence[int],
) -> ProgrammeEntier:
    """(SCP) : chaque objet au moins une fois, au moindre cout."""
    lignes = _incidences(objets, sous_ensembles, Sens.SUPERIEUR)
    return programme_binaire(Optimisation.MIN, couts, lignes)


def partition(
    objets: int,
    sous_ensembles: Sequence[frozenset[int]],
    couts: Sequence[int],
) -> ProgrammeEntier:
    """(SPP) : chaque objet exactement une fois, au moindre cout."""
    lignes = _incidences(objets, sous_ensembles, Sens.EGAL)
    return programme_binaire(Optimisation.MIN, couts, lignes)


def emballage(
    objets: int,
    sous_ensembles: Sequence[frozenset[int]],
    valeurs: Sequence[int],
) -> ProgrammeEntier:
    """« Set packing » : des sous-ensembles disjoints, de valeur maximale."""
    lignes = _incidences(objets, sous_ensembles, Sens.INFERIEUR)
    return programme_binaire(Optimisation.MAX, valeurs, lignes)
