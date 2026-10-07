"""Modelisations particulieres : les modeles des planches 48 a 56.

Chaque variante est ecrite comme sa planche l'ecrit, puis resolue. Le
terme constant `- 60` du stockage ne change pas l'optimum : il est
ajoute apres coup.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.modele_gare import en_entiers
from polytech.optimisation.application.modele_production import (
    COUTS,
    DEMANDES,
    avances_cumulees,
)
from polytech.optimisation.application.modele_variables import Variables, numerotees
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre

if TYPE_CHECKING:
    from collections.abc import Mapping

    import sympy as sp

    from polytech.optimisation.domain.algorithme import Issue

MOIS: Final = len(DEMANDES)
X: Final = numerotees("x", MOIS)
GRAND_M: Final = sum(DEMANDES)
"""Aucun mois ne produit plus que la demande totale."""
Termes = dict[str, int]


def production_et_stock() -> Termes:
    """`x1 + 4 x2 + ...` plus `4 x1 + 3 x2 + 2 x3 + x4` : le cout sans sa constante."""
    return {x: COUTS[i] + MOIS - 1 - i for i, x in enumerate(X)}


def socle(variables: Variables, decalages: Mapping[int, Termes]) -> list[Contrainte]:
    """Les demandes cumulees servies a temps, et la demande totale produite."""
    lignes = []
    cumulees = tuple(itertools.accumulate(DEMANDES))
    for mois, cumul in enumerate(cumulees, start=1):
        termes = dict.fromkeys(X[:mois], 1) | decalages.get(mois, {})
        sens = Sens.EGAL if mois == MOIS else Sens.SUPERIEUR
        lignes.append(Contrainte(variables.ligne(termes), sens, cumul))
    return lignes


def _extras(
    variante: str,
) -> tuple[tuple[str, ...], Termes, list[tuple[Termes, Sens, int]]]:
    """Les variables, les termes d'objectif et les contraintes propres a la variante."""
    paires = list(itertools.pairwise(X))
    if variante == "II":
        croissance = [({b: 1, a: -1}, Sens.SUPERIEUR, 0) for a, b in paires]
        return (), {"x1": -2, f"x{MOIS}": 2}, croissance
    if variante == "III":
        c = numerotees("c", MOIS - 1)
        triplets = [(k, a, b) for k, (a, b) in zip(c, paires, strict=True)]
        hausses = [({k: 1, b: -4, a: 4}, Sens.SUPERIEUR, 0) for k, a, b in triplets]
        baisses = [({k: 1, a: -2, b: 2}, Sens.SUPERIEUR, 0) for k, a, b in triplets]
        return c, dict.fromkeys(c, 1), hausses + baisses
    if variante == "IV":
        plus = [({"c1": 1, x: -1}, Sens.SUPERIEUR, 0) for x in X]
        moins = [({"c2": 1, x: -1}, Sens.INFERIEUR, 0) for x in X]
        return ("c1", "c2"), {"c1": 2, "c2": -2}, plus + moins
    if variante == "V":
        return (
            ("d",),
            {},
            [
                ({"x1": 1, "x2": -1}, Sens.INFERIEUR, 2),
                ({"x2": 1, "x1": -1}, Sens.INFERIEUR, 2),
                ({"x3": 1, "x4": -1, "d": GRAND_M}, Sens.SUPERIEUR, 3),
                ({"x4": 1, "x3": -1, "d": -GRAND_M}, Sens.SUPERIEUR, 3 - GRAND_M),
                ({"d": 1}, Sens.INFERIEUR, 1),
            ],
        )
    msg = f"variante sans modele ici : {variante}"
    raise KeyError(msg)


def programme(variante: str) -> ProgrammeGeneral:
    """Le modele de la planche de cette variante (I a V), sans le terme constant."""
    if variante == "I":
        couts = dict(zip(X, COUTS, strict=True))
        return ProgrammeGeneral(
            Optimisation.MIN, Variables(X).ligne(couts), tuple(socle(Variables(X), {}))
        )
    noms, objectif, propres = _extras(variante)
    variables = Variables((*X, *noms))
    contraintes = socle(variables, {})
    contraintes += [Contrainte(variables.ligne(t), s, d) for t, s, d in propres]
    cout = variables.ligne(
        production_et_stock() | _somme(production_et_stock(), objectif)
    )
    return ProgrammeGeneral(Optimisation.MIN, cout, tuple(contraintes))


def _somme(base: Termes, ajout: Termes) -> Termes:
    """Les termes de `ajout`, ajoutes a ceux de `base` qui portent le meme nom."""
    return {nom: base.get(nom, 0) + valeur for nom, valeur in ajout.items()}


def issue(variante: str) -> Issue:
    """Ce que le simplexe conclut sur le modele, variables continues."""
    return resoudre(programme(variante)).issue


def optimum(variante: str) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Le cout minimal en variables entieres, et la production des cinq mois."""
    valeur, point = en_entiers(programme(variante))
    constante = 0 if variante == "I" else avances_cumulees(DEMANDES)
    return valeur - constante, point[:MOIS]
