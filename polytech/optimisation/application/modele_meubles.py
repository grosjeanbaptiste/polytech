"""Modelisation, problemes VIa et VIb : huit semaines de meubles.

Deux meubles, « De Luxe » (`L`) et « Standard » (`S`), passent par la
fabrication (atelier `F1` ou `F2`), l'assemblage (`A1` ou `A2`) puis la
finition. On decide, semaine par semaine, ce qu'on produit dans chaque
atelier, ce qu'on vend et ce qu'on stocke.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.modele_variables import Variables
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Mapping

SEMAINES: Final = 8
MEUBLES: Final = ("L", "S")
MINUTES: Final = {
    "F1": {"L": 30, "S": 10},
    "F2": {"L": 20, "S": 30},
    "A1": {"L": 10, "S": 30},
    "A2": {"L": 15, "S": 20},
    "Fi": {"L": 20, "S": 10},
}
"""Planche 23 : le temps, en minutes, d'un meuble dans chaque atelier."""
CAPACITES: Final = {"F1": 7200, "F2": 6000, "A1": 7800, "A2": 6000, "Fi": 9000}
"""Les capacites hebdomadaires de la planche 23, converties en minutes."""
EXTENSIBLES: Final = ("F2", "A2")
VENTES: Final = {
    "L": (500, 350, 300, 450, 400, 470, 490, 475),
    "S": (300, 350, 600, 500, 300, 350, 410, 350),
}
GAINS: Final = {"L": 300, "S": 125}
STOCKAGE: Final = 25
STOCKS_DE_L_ENONCE: Final = {"L": 200, "S": 100}
"""Planche 23 : « un stock de 100 Standard et 200 De Luxe »."""
STOCKS_IMPRIMES: Final = {"L": 100, "S": 200}
"""Planche 24 : les deux derniers bilans portent les stocks echanges."""
ETAPES: Final = (("F1", "F2"), ("A1", "A2"))


def _noms() -> tuple[str, ...]:
    noms = []
    for t in range(1, SEMAINES + 1):
        noms += [f"P_{m}_{a}_{t}" for m in MEUBLES for etape in ETAPES for a in etape]
        noms += [f"V_{m}_{t}" for m in MEUBLES]
        noms += [f"S_{m}_{t}" for m in MEUBLES if t < SEMAINES]
    return tuple(noms)


VARIABLES: Final = Variables(_noms())


def capacite(atelier: str, semaine: int, extension: int | None) -> int:
    """La capacite de l'atelier, doublee a partir de la semaine d'extension."""
    etendu = extension is not None and semaine >= extension
    return CAPACITES[atelier] * (2 if etendu and atelier in EXTENSIBLES else 1)


def _ateliers(t: int, extension: int | None) -> list[Contrainte]:
    """Les cinq capacites de la semaine, et le passage par les deux etapes."""
    ligne = VARIABLES.ligne
    lignes = []
    for atelier in (*ETAPES[0], *ETAPES[1]):
        termes = {f"P_{m}_{atelier}_{t}": MINUTES[atelier][m] for m in MEUBLES}
        borne = capacite(atelier, t, extension)
        lignes.append(Contrainte(ligne(termes), Sens.INFERIEUR, borne))
    finition = {f"P_{m}_{a}_{t}": MINUTES["Fi"][m] for m in MEUBLES for a in ETAPES[0]}
    lignes.append(Contrainte(ligne(finition), Sens.INFERIEUR, CAPACITES["Fi"]))
    for m in MEUBLES:
        passage = {f"P_{m}_{a}_{t}": 1 for a in ETAPES[0]}
        passage |= {f"P_{m}_{a}_{t}": -1 for a in ETAPES[1]}
        lignes.append(Contrainte(ligne(passage), Sens.EGAL, 0))
    return lignes


def _bilans(t: int, stocks: Mapping[str, int]) -> list[Contrainte]:
    """Production plus stock entrant = ventes plus stock sortant ; ventes plafonnees."""
    ligne = VARIABLES.ligne
    lignes = []
    for m in MEUBLES:
        bilan = {f"P_{m}_{a}_{t}": 1 for a in ETAPES[0]} | {f"V_{m}_{t}": -1}
        if t > 1:
            bilan[f"S_{m}_{t - 1}"] = 1
        if t < SEMAINES:
            bilan[f"S_{m}_{t}"] = -1
        final = stocks[m] if t == SEMAINES else 0
        lignes.append(Contrainte(ligne(bilan), Sens.EGAL, final))
        plafond = VENTES[m][t - 1]
        lignes.append(Contrainte(ligne({f"V_{m}_{t}": 1}), Sens.INFERIEUR, plafond))
    return lignes


def programme(
    stocks: Mapping[str, int], extension: int | None = None
) -> ProgrammeGeneral:
    """Le modele de la planche 24 ; avec `extension`, les capacites de la planche 26."""
    objectif: dict[str, int] = {}
    contraintes: list[Contrainte] = []
    for t in range(1, SEMAINES + 1):
        objectif |= {f"V_{m}_{t}": GAINS[m] for m in MEUBLES}
        if t < SEMAINES:
            objectif |= {f"S_{m}_{t}": -STOCKAGE for m in MEUBLES}
        contraintes += _ateliers(t, extension) + _bilans(t, stocks)
    return ProgrammeGeneral(
        Optimisation.MAX, VARIABLES.ligne(objectif), tuple(contraintes)
    )
