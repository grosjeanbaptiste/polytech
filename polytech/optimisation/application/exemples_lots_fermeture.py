"""Exemples resolus, planches 84 et 85 : la fermeture convexe du « lot-sizing ».

`w_it` est ce que le mois `i` produit pour la demande du mois `t`. Chaque
`w_it` est borne par `d_t y_i` : la relaxation n'a plus de marge a
prendre, et son optimum est entier.
"""

from __future__ import annotations

from typing import Final

from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

DEMANDES: Final = (3, 5, 6, 3, 8)
UNITAIRES: Final = (2, 4, 6, 8, 10)
STOCKAGES: Final = (3, 2, 3, 2)
FORFAITS: Final = (10, 8, 6, 4, 2)
"""Planche 80 : par mois, la demande, le cout unitaire, le stockage, le forfait."""
MOIS: Final = len(DEMANDES)
COUPLES: Final = tuple((i, t) for i in range(MOIS) for t in range(i, MOIS))
"""Les quinze `w_it` : on ne produit pas pour un mois passe."""


def cout_d_une_unite(produite: int, servie: int) -> int:
    """Ce que coute une unite produite un mois et stockee jusqu'a un autre."""
    return UNITAIRES[produite] + sum(STOCKAGES[produite:servie])


def _ligne(termes: dict[int, int]) -> tuple[int, ...]:
    ligne = [0] * (len(COUPLES) + MOIS)
    for rang, coefficient in termes.items():
        ligne[rang] = coefficient
    return tuple(ligne)


def fermeture_convexe() -> ProgrammeGeneral:
    """L'ecriture IV : quinze `w`, puis les cinq `y`, relaches entre 0 et 1."""
    objectif = (*(cout_d_une_unite(i, t) for i, t in COUPLES), *FORFAITS)
    demandes = [
        Contrainte(
            _ligne({k: 1 for k, (_, servi) in enumerate(COUPLES) if servi == t}),
            Sens.EGAL,
            DEMANDES[t],
        )
        for t in range(MOIS)
    ]
    liens = [
        Contrainte(_ligne({k: 1, len(COUPLES) + i: -DEMANDES[t]}), Sens.INFERIEUR, 0)
        for k, (i, t) in enumerate(COUPLES)
    ]
    unites = [
        Contrainte(_ligne({len(COUPLES) + i: 1}), Sens.INFERIEUR, 1)
        for i in range(MOIS)
    ]
    contraintes = (*demandes, *liens, *unites)
    return ProgrammeGeneral(Optimisation.MIN, objectif, contraintes)
