"""Bernoulli instationnaire : le terme `integrale de dV/dt ds`.

Le long d'une ligne de courant, a **un meme instant**, entre deux points :
`p_1 + rho g z_1 + rho V_1^2/2 + rho I = p_2 + rho g z_2 + rho V_2^2/2`
avec `I = integrale_1^2 (dV/dt) ds`. Le temps n'entre que par cette
acceleration locale ; Bernoulli ne relie jamais deux instants. Pour un
fluide incompressible dans une conduite, `V = Q(t) / A(s)` et donc
`I = (dQ/dt) somme(L_i / A_i)` sur les troncons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.fluides.domain.grandeurs import PESANTEUR

if TYPE_CHECKING:
    from collections.abc import Sequence


def terme_instationnaire(
    variation_du_debit: float, troncons: Sequence[tuple[float, float]]
) -> float:
    """`(dQ/dt) somme(L_i / A_i)` ; chaque troncon est `(longueur, section)`."""
    if not troncons or any(longueur <= 0 or aire <= 0 for longueur, aire in troncons):
        msg = f"troncon absent ou degenere (longueur, section) : {list(troncons)}"
        raise ValueError(msg)
    return variation_du_debit * sum(longueur / aire for longueur, aire in troncons)


def pression_en_aval_instationnaire(  # noqa: PLR0913 -- les sept grandeurs de l'equation
    pression_amont: float,
    masse_volumique: float,
    vitesse_amont: float,
    vitesse_aval: float,
    altitude_amont: float,
    altitude_aval: float,
    terme: float,
) -> float:
    """`p_2 = p_1 + rho (V_1^2 - V_2^2)/2 + rho g (z_1 - z_2) - rho I`."""
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)
    return (
        pression_amont
        + masse_volumique * (vitesse_amont**2 - vitesse_aval**2) / 2
        + masse_volumique * PESANTEUR * (altitude_amont - altitude_aval)
        - masse_volumique * terme
    )
