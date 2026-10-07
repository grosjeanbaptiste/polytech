"""Modelisation, probleme VIa : une semaine d'atelier, vue seule.

Second chemin pour relire le plan : deux quantites sont-elles
fabricables en une semaine ? On les repartit entre les deux ateliers de
fabrication et les deux d'assemblage, sous les cinq capacites.
"""

from __future__ import annotations

from polytech.optimisation.application.modele_meubles import (
    CAPACITES,
    ETAPES,
    MEUBLES,
    MINUTES,
)
from polytech.optimisation.application.modele_variables import Variables
from polytech.optimisation.domain.algorithme import Issue
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre

_VARIABLES = Variables(tuple(f"{m}_{a}" for m in MEUBLES for e in ETAPES for a in e))


def une_semaine(de_luxe: int, standard: int) -> bool:
    """Vrai si les cinq ateliers peuvent sortir ces quantites en une semaine."""
    quantites = dict(zip(MEUBLES, (de_luxe, standard), strict=True))
    finition = sum(MINUTES["Fi"][m] * quantites[m] for m in MEUBLES)
    if finition > CAPACITES["Fi"]:
        return False
    ligne = _VARIABLES.ligne
    contraintes = [
        Contrainte(ligne({f"{m}_{a}": 1 for a in etape}), Sens.EGAL, quantites[m])
        for m in MEUBLES
        for etape in ETAPES
    ]
    for etape in ETAPES:
        for atelier in etape:
            charge = {f"{m}_{atelier}": MINUTES[atelier][m] for m in MEUBLES}
            contraintes.append(
                Contrainte(ligne(charge), Sens.INFERIEUR, CAPACITES[atelier])
            )
    objectif = (0,) * len(_VARIABLES.noms)
    modele = ProgrammeGeneral(Optimisation.MIN, objectif, tuple(contraintes))
    return resoudre(modele).issue is Issue.OPTIMALE
