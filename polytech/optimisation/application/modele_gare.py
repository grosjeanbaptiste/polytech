"""Modelisation, problemes IIIa et IIIb : les equipes d'une gare.

La journee compte six periodes de quatre heures. Une equipe prend son
service au debut d'une periode et travaille huit heures : elle couvre
sa periode et la suivante. `x_i` est l'effectif de l'equipe qui commence
a la periode `i`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy as sp

PERIODES: Final = 6
BESOINS: Final = (3, 8, 10, 8, 14, 5)
"""Planche 12 : de 0 h - 4 h a 20 h - 24 h."""
VARIANTES: Final = ((4, 9, 9, 9, 12, 7), (2, 10, 12, 12, 12, 4))
"""Planche 15 : deux autres definitions des besoins."""
NUIT: Final = frozenset({6, 1, 2})
"""De 20 h a 8 h : ces periodes sont payees double."""
ORDRE_DES_LIGNES: Final = (2, 3, 4, 5, 6, 1)
"""Les planches ecrivent la contrainte de la periode 1 en dernier."""
Entiers = tuple["sp.Expr", ...]


def periodes_de(debut: int) -> tuple[int, int]:
    """Les deux periodes que couvre l'equipe qui commence a celle-ci."""
    return debut, debut % PERIODES + 1


def heures_payees(debut: int) -> int:
    """Quatre heures par periode, huit si elle est de nuit."""
    return sum(8 if periode in NUIT else 4 for periode in periodes_de(debut))


SALAIRES: Final = tuple(heures_payees(debut) for debut in range(1, PERIODES + 1))


def presents(equipes: Sequence[sp.Expr | int]) -> tuple[sp.Expr | int, ...]:
    """Le nombre de personnes au travail pendant chaque periode."""
    return tuple(
        sum(n for debut, n in enumerate(equipes, start=1) if p in periodes_de(debut))
        for p in range(1, PERIODES + 1)
    )


def est_couvert(besoins: Sequence[int], equipes: Sequence[sp.Expr | int]) -> bool:
    """Vrai si chaque periode a au moins l'effectif demande."""
    return all(p >= b for p, b in zip(presents(equipes), besoins, strict=True))


def besoin_des_periodes(besoins: Sequence[int], periodes: Sequence[int]) -> int:
    """La somme des besoins de periodes qu'aucune equipe ne couvre a deux."""
    return sum(besoins[p - 1] for p in periodes)


def ligne_de_presence(periode: int, variables: int = PERIODES) -> list[int]:
    """Les coefficients des equipes presentes a cette periode."""
    return [
        int(debut <= PERIODES and periode in periodes_de(debut))
        for debut in range(1, variables + 1)
    ]


def programme(besoins: Sequence[int], objectif: str) -> ProgrammeGeneral:
    """Le modele de la planche 13 (`effectif`) ou de la planche 14 (`salaire`)."""
    objectifs = {"effectif": (1,) * PERIODES, "salaire": SALAIRES}
    if objectif not in objectifs:
        msg = f"objectif inconnu : {objectif}"
        raise KeyError(msg)
    contraintes = tuple(
        Contrainte(tuple(ligne_de_presence(p)), Sens.SUPERIEUR, besoins[p - 1])
        for p in ORDRE_DES_LIGNES
    )
    return ProgrammeGeneral(Optimisation.MIN, objectifs[objectif], contraintes)


def en_entiers(modele: ProgrammeGeneral) -> tuple[sp.Expr, Entiers]:
    """L'optimum en variables toutes entieres ; leve s'il n'y en a pas."""
    entieres = frozenset(range(len(modele.objectif)))
    exploration = resoudre_en_variables_entieres(ProgrammeEntier(modele, entieres))
    if exploration.valeur is None or exploration.solution is None:
        msg = "aucune solution entiere : le modele est impossible"
        raise ValueError(msg)
    return exploration.valeur, tuple(exploration.solution)


def optimum(besoins: Sequence[int], objectif: str) -> tuple[sp.Expr, Entiers]:
    """Le minimum de l'objectif, et les six effectifs qui l'atteignent."""
    return en_entiers(programme(besoins, objectif))


def ecarts(variante: Sequence[int]) -> tuple[int, ...]:
    """Ce que la variante ajoute a chaque besoin, dans l'ordre des lignes."""
    return tuple(variante[p - 1] - BESOINS[p - 1] for p in ORDRE_DES_LIGNES)


def cout_par_definition() -> tuple[sp.Expr, ...]:
    """Le cout salarial minimal pour chacune des trois definitions des besoins."""
    return tuple(optimum(b, "salaire")[0] for b in (BESOINS, *VARIANTES))


def choix_du_modele() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Planche 16 : le cout, `delta` et `gamma` du modele a deux binaires."""
    lignes = []
    for p, e_delta, e_gamma in zip(
        ORDRE_DES_LIGNES, ecarts(VARIANTES[0]), ecarts(VARIANTES[1]), strict=True
    ):
        ligne = (*ligne_de_presence(p), -e_delta, -e_gamma)
        lignes.append(Contrainte(ligne, Sens.SUPERIEUR, BESOINS[p - 1]))
    un_seul = Contrainte((*(0,) * PERIODES, 1, 1), Sens.INFERIEUR, 1)
    modele = ProgrammeGeneral(Optimisation.MIN, (*SALAIRES, 0, 0), (*lignes, un_seul))
    cout, point = en_entiers(modele)
    return cout, point[PERIODES], point[PERIODES + 1]
