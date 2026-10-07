"""Modelisations particulieres : les modeles a variables binaires (planches 58 a 69).

Une variable binaire `d_i` dit si l'on fabrique le mois `i` ; un grand
`M` la lie a la quantite. Les variantes VI a X posent des conditions
logiques sur ces `d_i` ; la variante XI choisit entre deux demandes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.modele_gare import en_entiers
from polytech.optimisation.application.modele_production import (
    ALTERNATIVE,
    DEMANDES,
    avances_cumulees,
)
from polytech.optimisation.application.modele_production_planches import (
    GRAND_M,
    MOIS,
    Termes,
    X,
    production_et_stock,
    socle,
)
from polytech.optimisation.application.modele_variables import Variables, numerotees
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

D: Final = numerotees("d", MOIS)
Ligne = tuple[Termes, Sens, int]
INF, SUP, EGAL = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL


def _liens(mois: int, *, reciproque: bool) -> list[Ligne]:
    """`x_i <= M d_i`, `d_i <= 1` et, si demande, `d_i <= x_i`."""
    lignes: list[Ligne] = []
    for x, d in zip(X[:mois], D[:mois], strict=True):
        lignes += [({x: 1, d: -GRAND_M}, INF, 0), ({d: 1}, INF, 1)]
        if reciproque:
            lignes.append(({d: 1, x: -1}, INF, 0))
    return lignes


def _propres(variante: str) -> list[Ligne]:
    """Les contraintes de la planche, au-dela des demandes."""
    tous = dict.fromkeys(D, 1)
    if variante == "VI":
        return [*_liens(MOIS, reciproque=False), (tous, INF, 3)]
    if variante == "VII":
        return [*_liens(MOIS, reciproque=True), (tous, EGAL, 3)]
    if variante == "VIII":
        implication = {"d4": 1, "d1": -1, "d2": -1, "d3": -1}
        return [*_liens(4, reciproque=True), (implication, SUP, -2)]
    if variante == "IX":
        return [({"x4": 1, x: -GRAND_M}, INF, 0) for x in X[:3]]
    if variante == "X":
        return [({"x4": 1, "x2": -GRAND_M, "x3": -GRAND_M}, INF, 0)]
    msg = f"variante sans modele ici : {variante}"
    raise KeyError(msg)


def programme(variante: str) -> ProgrammeGeneral:
    """Le modele de la planche de cette variante (VI a X), sans le terme constant."""
    variables = Variables((*X, *D))
    contraintes = socle(variables, {})
    contraintes += [
        Contrainte(variables.ligne(t), s, d) for t, s, d in _propres(variante)
    ]
    objectif = variables.ligne(production_et_stock())
    return ProgrammeGeneral(Optimisation.MIN, objectif, tuple(contraintes))


def optimum(variante: str) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Le cout minimal, et la production des cinq mois."""
    valeur, point = en_entiers(programme(variante))
    return valeur - avances_cumulees(DEMANDES), point[:MOIS]


def choix_de_la_demande(
    *, corrige: bool, impose: int | None = None
) -> tuple[sp.Expr, sp.Expr, tuple[sp.Expr, ...]]:
    """Planche 69 : le cout, `delta` et le plan.

    `corrige` retablit le stock exact ; `impose` fixe `delta` d'autorite.
    """
    variables = Variables((*X, "d"))
    ecarts = [
        a - b for a, b in zip(_cumulees(ALTERNATIVE), _cumulees(DEMANDES), strict=True)
    ]
    decalages = {mois: {"d": -e} for mois, e in enumerate(ecarts, start=1) if e}
    contraintes = [
        *socle(variables, decalages),
        Contrainte(variables.ligne({"d": 1}), INF, 1),
    ]
    if impose is not None:
        contraintes.append(Contrainte(variables.ligne({"d": 1}), EGAL, impose))
    oubli = avances_cumulees(DEMANDES) - avances_cumulees(ALTERNATIVE)
    objectif = production_et_stock() | ({"d": oubli} if corrige else {})
    modele = ProgrammeGeneral(
        Optimisation.MIN, variables.ligne(objectif), tuple(contraintes)
    )
    valeur, point = en_entiers(modele)
    return valeur - avances_cumulees(DEMANDES), point[MOIS], point[:MOIS]


def _cumulees(demandes: tuple[int, ...]) -> list[int]:
    cumul, cumulees = 0, []
    for demande in demandes:
        cumul += demande
        cumulees.append(cumul)
    return cumulees
