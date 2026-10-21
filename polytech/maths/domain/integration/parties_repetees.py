"""L'integration par parties repetee, telle que les notes de 2011 la menent.

Chaque etape reprend l'integrale restante `int v du` avec `u <- du` et
`dv <- v`. Deux fins (notes, pages 74 et 75) :
- **epuisement** : `du` ne depend plus de `x` ; l'integrale restante,
  `du . int v`, se primitive a vue -- `(x + 1)^3 ch x` en trois parties ;
- **boucle** : l'integrale restante est `c` fois celle de depart, `I` ;
  on « fait passer » ce terme dans le membre de gauche :
  `(1 - s c) I = S`, `s` etant le signe qu'elle porte -- `2^x sin 3x`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.symboles import VARIABLE

MAXIMUM = 12
"""Les notes en font trois ; au-dela de douze, la suite ne s'acheve pas."""


@dataclass(frozen=True, slots=True)
class PartiesRepetees:
    """Les etapes, dans l'ordre, la primitive, et la fin rencontree."""

    etapes: tuple[IntegrationParParties, ...]
    primitive: sp.Expr
    en_boucle: bool


def jusqu_au_bout(
    u: sp.Expr, derivee_de_v: sp.Expr, *, maximum: int = MAXIMUM
) -> PartiesRepetees:
    """Iterer le geste jusqu'a epuisement ou jusqu'a retomber sur l'integrale."""
    depart = u * derivee_de_v
    etapes = [IntegrationParParties(u, derivee_de_v)]
    somme, signe = sp.Integer(0), 1
    while len(etapes) <= maximum:
        etape = etapes[-1]
        somme += signe * etape.u * etape.v()
        signe = -signe
        restante = etape.v() * etape.derivee_de_u()
        if not etape.derivee_de_u().has(VARIABLE):
            primitive = somme + signe * sp.integrate(restante, VARIABLE)
            return PartiesRepetees(
                tuple(etapes), sp.simplify(primitive), en_boucle=False
            )
        rapport = sp.simplify(restante / depart)
        if not rapport.has(VARIABLE):
            primitive = somme / (1 - signe * rapport)
            return PartiesRepetees(
                tuple(etapes), sp.simplify(primitive), en_boucle=True
            )
        etapes.append(IntegrationParParties(etape.derivee_de_u(), etape.v()))
    msg = f"l'integration par parties de {depart} ne s'acheve pas en {maximum} etapes"
    raise ValueError(msg)
