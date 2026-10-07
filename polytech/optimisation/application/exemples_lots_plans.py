"""Exemples resolus, planches 79 a 85 : le « lot-sizing » plan par plan.

Second chemin vers l'optimum : au lieu d'un programme lineaire, on
chiffre directement chaque plan de production possible.
"""

from __future__ import annotations

import itertools
from typing import Final

import sympy as sp

from polytech.optimisation.application.exemples_lots_fermeture import (
    DEMANDES,
    FORFAITS,
    STOCKAGES,
    UNITAIRES,
)

MOIS: Final = len(DEMANDES)


def cout_du_plan(production: tuple[int, ...]) -> sp.Expr:
    """Le cout d'un plan de production, les stocks suivant des bilans."""
    stock, total = 0, sp.Integer(0)
    for mois, (produit, demande) in enumerate(zip(production, DEMANDES, strict=True)):
        stock += produit - demande
        if stock < 0:
            msg = f"le plan ne sert pas la demande du mois {mois + 1}"
            raise ValueError(msg)
        total += UNITAIRES[mois] * produit + (FORFAITS[mois] if produit else 0)
        total += STOCKAGES[mois] * stock if mois < MOIS - 1 else 0
    if stock:
        msg = f"le plan laisse {stock} unites apres le dernier mois"
        raise ValueError(msg)
    return total


def plan_des_mois_ouverts(ouverts: tuple[int, ...]) -> tuple[int, ...]:
    """Chaque demande est produite au dernier mois ouvert qui la precede.

    Les couts unitaires croissent de 2 par mois et stocker coute au moins
    2 : produire plus tot ne fait jamais gagner.
    """
    plan = [0] * MOIS
    for mois, demande in enumerate(DEMANDES):
        plan[max(i for i in range(mois + 1) if ouverts[i])] += demande
    return tuple(plan)


def optimum_par_enumeration() -> tuple[sp.Expr, tuple[int, ...], tuple[int, ...]]:
    """Le moins cher des seize choix de mois ouverts ; le premier l'est toujours."""
    choix = [(1, *reste) for reste in itertools.product((0, 1), repeat=MOIS - 1)]
    plans = [(cout_du_plan(plan_des_mois_ouverts(o)), o) for o in choix]
    valeur, ouverts = min(plans)
    return valeur, ouverts, plan_des_mois_ouverts(ouverts)
