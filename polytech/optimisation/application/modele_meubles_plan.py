"""Modelisation, problemes VIa et VIb : le plan optimal, et ce qu'il enseigne.

Le simplexe rend un plan ; ce module le relit sur l'enonce, et mesure ce
que l'extension de capacite de la planche 25 rapporterait.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.application.modele_meubles import (
    CAPACITES,
    ETAPES,
    EXTENSIBLES,
    GAINS,
    MEUBLES,
    MINUTES,
    SEMAINES,
    STOCKAGE,
    STOCKS_IMPRIMES,
    VARIABLES,
    VENTES,
    programme,
)
from polytech.optimisation.application.optimum import optimum_fini

if TYPE_CHECKING:
    from collections.abc import Mapping

COUTS_FIXES: Final = {6: 5000, 4: 7500}
"""Planche 25 : doubler `F2` et `A2` des la semaine 6, ou des la semaine 4."""
Stocks = tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class Plan:
    """Le gain et, par meuble, la production et les ventes de chaque semaine."""

    gain: sp.Expr
    production: Mapping[str, tuple[sp.Expr, ...]]
    ventes: Mapping[str, tuple[sp.Expr, ...]]


@cache
def plan(stocks: Stocks, extension: int | None = None) -> Plan:
    """Un plan de gain maximal, par le simplexe."""
    gain, point = optimum_fini(programme(dict(stocks), extension))
    valeurs = dict(zip(VARIABLES.noms, point, strict=True))
    production = {
        m: tuple(
            sum((valeurs[f"P_{m}_{a}_{t}"] for a in ETAPES[0]), sp.Integer(0))
            for t in range(1, SEMAINES + 1)
        )
        for m in MEUBLES
    }
    ventes = {m: VARIABLES.lire(point, f"V_{m}_") for m in MEUBLES}
    return Plan(gain, production, ventes)


def stocks_successifs(le_plan: Plan, meuble: str) -> tuple[sp.Expr, ...]:
    """Le stock a la fin de chaque semaine : ce qui est produit et pas vendu."""
    niveau, niveaux = sp.Integer(0), []
    for produit, vendu in zip(
        le_plan.production[meuble], le_plan.ventes[meuble], strict=True
    ):
        niveau += produit - vendu
        niveaux.append(niveau)
    return tuple(niveaux)


def stock_final(le_plan: Plan) -> dict[str, sp.Expr]:
    """Le stock de chaque meuble au debut de la semaine 9."""
    return {m: stocks_successifs(le_plan, m)[-1] for m in MEUBLES}


def ventes_dans_les_previsions(le_plan: Plan) -> bool:
    """Vrai si aucune vente ne depasse la prevision ni ne puise dans un stock vide."""
    plafonnees = all(
        0 <= v <= p
        for m in MEUBLES
        for v, p in zip(le_plan.ventes[m], VENTES[m], strict=True)
    )
    return plafonnees and all(
        s >= 0 for m in MEUBLES for s in stocks_successifs(le_plan, m)
    )


def gain_relu(le_plan: Plan) -> sp.Expr:
    """Les ventes moins le stockage des sept premieres semaines, d'apres l'enonce."""
    ventes = sum(GAINS[m] * sum(le_plan.ventes[m]) for m in MEUBLES)
    stockes = sum(sum(stocks_successifs(le_plan, m)[:-1]) for m in MEUBLES)
    return ventes - STOCKAGE * stockes


def _plafond_d_une_etape(
    etape: tuple[str, ...], meuble: str, *, etendu: bool
) -> sp.Expr:
    """Ce que les ateliers d'une etape sortent en une semaine, ce meuble seul."""
    total = sp.Integer(0)
    for atelier in etape:
        double = etendu and atelier in EXTENSIBLES
        minutes = CAPACITES[atelier] * (2 if double else 1)
        total += sp.Rational(minutes, MINUTES[atelier][meuble])
    return total


def plafond_hebdomadaire(meuble: str, *, etendu: bool = False) -> sp.Expr:
    """Le plus de meubles de ce type en une semaine : l'etape la plus lente decide."""
    etapes = [_plafond_d_une_etape(e, meuble, etendu=etendu) for e in ETAPES]
    finition = sp.Rational(CAPACITES["Fi"], MINUTES["Fi"][meuble])
    return sp.floor(min(*etapes, finition))


def minutes_de_finition(le_plan: Plan) -> tuple[sp.Expr, ...]:
    """Le temps de finition que la production de chaque semaine consomme."""
    return tuple(
        sum(MINUTES["Fi"][m] * le_plan.production[m][t] for m in MEUBLES)
        for t in range(SEMAINES)
    )


def gain_net_de_l_extension(semaine: int) -> sp.Expr:
    """Ce que rapporte l'extension decidee cette semaine-la, cout fixe deduit."""
    if semaine not in COUTS_FIXES:
        msg = f"aucune extension prevue la semaine {semaine}"
        raise KeyError(msg)
    stocks = tuple(STOCKS_IMPRIMES.items())
    return plan(stocks, semaine).gain - plan(stocks).gain - COUTS_FIXES[semaine]
