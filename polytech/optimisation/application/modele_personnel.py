"""Modelisation, probleme I : le personnel de bord d'une compagnie aerienne.

Douze mois. `T_i` stewards travaillent le mois `i` ; on en embauche `E_i`
et on en licencie `L_i` en debut de mois, cinq partent d'eux-memes a la
fin. `H_i` compte les heures supplementaires. Le treizieme indice est le
debut du mois de janvier suivant, ou l'effectif doit revenir a 60.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.modele_variables import Variables, numerotees
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

BESOINS: Final = (
    7500,
    9000,
    10500,
    9000,
    6000,
    7200,
    12000,
    15000,
    10500,
    12000,
    9000,
    10500,
)
"""Planche 4 : les heures de travail a fournir, de janvier a decembre."""
MOIS: Final = len(BESOINS)
DEPART, DEPARTS = 60, 5
HEURES, HEURES_SUP = 150, 20
SALAIRE, INDEMNITE, COUT_HEURE_SUP = 3000, 3500, 50
VARIABLES: Final = Variables(
    (
        *numerotees("T", MOIS),
        *numerotees("E", MOIS + 1),
        *numerotees("L", MOIS + 1),
        *numerotees("H", MOIS),
    )
)


@dataclass(frozen=True, slots=True)
class Plan:
    """Le cout total et, mois par mois, ce que le plan decide."""

    cout: sp.Expr
    effectifs: tuple[sp.Expr, ...]
    embauches: tuple[sp.Expr, ...]
    licenciements: tuple[sp.Expr, ...]
    heures_supplementaires: tuple[sp.Expr, ...]


def _bilans() -> list[Contrainte]:
    """`T_i = T_(i-1) + E_i - L_i - 5`, de 60 au depart a 60 a l'arrivee."""
    ligne = VARIABLES.ligne
    bilans = [Contrainte(ligne({"T1": 1, "E1": -1, "L1": 1}), Sens.EGAL, DEPART)]
    for i in range(2, MOIS + 1):
        termes = {f"T{i}": 1, f"T{i - 1}": -1, f"E{i}": -1, f"L{i}": 1}
        bilans.append(Contrainte(ligne(termes), Sens.EGAL, -DEPARTS))
    fin = {f"T{MOIS}": 1, f"E{MOIS + 1}": 1, f"L{MOIS + 1}": -1}
    return [*bilans, Contrainte(ligne(fin), Sens.EGAL, DEPART + DEPARTS)]


def programme() -> ProgrammeGeneral:
    """Le modele de la planche 5."""
    ligne = VARIABLES.ligne
    objectif: dict[str, int] = {f"L{MOIS + 1}": INDEMNITE}
    contraintes = _bilans()
    for i, besoin in enumerate(BESOINS, start=1):
        objectif |= {f"T{i}": SALAIRE, f"L{i}": INDEMNITE, f"H{i}": COUT_HEURE_SUP}
        travail = ligne({f"T{i}": HEURES, f"H{i}": 1})
        plafond = ligne({f"H{i}": 1, f"T{i}": -HEURES_SUP})
        contraintes.append(Contrainte(travail, Sens.SUPERIEUR, besoin))
        contraintes.append(Contrainte(plafond, Sens.INFERIEUR, 0))
    return ProgrammeGeneral(Optimisation.MIN, ligne(objectif), tuple(contraintes))


@cache
def plan() -> Plan:
    """Un plan de cout minimal, par le simplexe."""
    cout, point = optimum_fini(programme())
    lire = VARIABLES.lire
    return Plan(
        cout, lire(point, "T"), lire(point, "E"), lire(point, "L"), lire(point, "H")
    )


def _cout_du_mois(effectif: int, besoin: int) -> int | None:
    """Salaires et heures supplementaires du mois ; rien si le besoin n'est pas tenu."""
    manque = max(0, besoin - HEURES * effectif)
    if manque > HEURES_SUP * effectif:
        return None
    return SALAIRE * effectif + COUT_HEURE_SUP * manque


@cache
def _suite(mois: int, presents: int) -> int:
    """Le cout minimal a partir de ce mois, `presents` stewards etant deja la."""
    if mois == MOIS:
        return INDEMNITE * max(0, presents - DEPART)
    totaux = []
    for effectif in range(max(BESOINS) // HEURES + 1):
        cout = _cout_du_mois(effectif, BESOINS[mois])
        if cout is not None:
            suite = _suite(mois + 1, effectif - DEPARTS)
            totaux.append(cout + INDEMNITE * max(0, presents - effectif) + suite)
    return min(totaux)


def cout_par_recurrence() -> int:
    """Le cout minimal, mois par mois, sur tous les effectifs entiers."""
    return _suite(0, DEPART)
