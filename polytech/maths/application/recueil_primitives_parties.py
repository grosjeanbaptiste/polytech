"""Les integrations par parties du recueil de primitives.

`u v - integrale de v du` : l'integrale qui reste est montree simplifiee,
parce que c'est sa forme simplifiee que l'on primitive ensuite. Au 54 de
l'exercice 1, `v du` vaut `x / racine(1 + x^2)` et non le quotient a deux
radicaux que la derivation donne d'abord.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application.recueil_primitives_gestes import valeur_absolue
from polytech.maths.application.resolution_exs_01 import Resolution
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.symboles import VARIABLE as X

PAR_PARTIES: Final = "integration par parties"


def _etape(u: sp.Expr, dv: sp.Expr) -> sp.Expr:
    """`u v - integrale de v du`, l'integrale qui reste etant simplifiee."""
    geste = IntegrationParParties(u, dv)
    restante = sp.simplify(geste.v() * geste.derivee_de_u())
    return geste.u * geste.v() - sp.Integral(restante, X)


def par_parties(u: sp.Expr, dv: sp.Expr) -> Resolution:
    """Une integration par parties, l'integrale restante se calculant directement."""
    etape = _etape(u, dv)
    return Resolution(u * dv, PAR_PARTIES, (etape,), valeur_absolue(etape.doit()))


def en_cascade(us: tuple[sp.Expr, ...], dv: sp.Expr) -> Resolution:
    """Des parties en chaine : chaque `u` est celui de l'integrale qui restait."""
    etapes = tuple(_etape(u, dv) for u in us)
    primitive = sp.expand(sp.integrate(us[0] * dv, X))
    return Resolution(us[0] * dv, PAR_PARTIES, etapes, primitive)
