"""Chapitre IV.3 : l'interpretation economique de la dualite.

Dans le primal `max c x`, `T x <= d`, les `d_i` sont des quantites de
biens disponibles et les `c_j` des gains unitaires. La variable duale
`u_i` est alors une **valeur imputee a une unite du bien `i`** -- un prix
fictif (« shadow price »), non un prix de marche :

* un bien en surplus a l'optimum a un prix fictif nul ;
* sinon, une unite de plus (ou de moins) de `d_i` deplace l'optimum de
  `u_i` : c'est la **valeur marginale** du bien, pour ce probleme.

La fonction lagrangienne `phi(x, u) = c x + u (d - T x)` reunit les deux
points de vue : l'entreprise la maximise en `x`, le marche la minimise en
`u`, et leur equilibre est l'optimum commun des deux problemes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.dual import dual_de, sous_la_convention
from polytech.optimisation.domain.resolution import resoudre

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.formulation import ProgrammeGeneral


def _optimum(programme: ProgrammeGeneral) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    solution = resoudre(programme)
    if solution.valeur is None or solution.point is None:
        msg = f"pas de solution optimale finie : {solution.issue.value}"
        raise ValueError(msg)
    return solution.valeur, solution.point


def prix_fictifs(primal: ProgrammeGeneral) -> tuple[sp.Expr, ...]:
    """La solution optimale du dual : un prix par contrainte du primal."""
    return _optimum(dual_de(primal))[1]


def valeur_d_une_unite_de_plus(primal: ProgrammeGeneral, contrainte: int) -> sp.Expr:
    """Ce que l'optimum gagne si `d_i` augmente de 1 (contrainte de 1 a `m`)."""
    visee = primal.contraintes[contrainte - 1]
    plus = replace(visee, second_membre=visee.second_membre + 1)
    contraintes = (
        *primal.contraintes[: contrainte - 1],
        plus,
        *primal.contraintes[contrainte:],
    )
    elargi = replace(primal, contraintes=contraintes)
    return sp.simplify(_optimum(elargi)[0] - _optimum(primal)[0])


def lagrangien(
    primal: ProgrammeGeneral,
    x: Sequence[object],
    u: Sequence[object],
) -> sp.Expr:
    """`phi(x, u) = c x + u (d - T x)`, le primal ecrit sous la convention."""
    convention = sous_la_convention(primal)
    residus = (c.second_membre - c.premier_membre(x) for c in convention.contraintes)
    penalite = sp.Add(
        *(sp.sympify(prix) * residu for prix, residu in zip(u, residus, strict=True))
    )
    return sp.simplify(primal.valeur(x) + penalite)
