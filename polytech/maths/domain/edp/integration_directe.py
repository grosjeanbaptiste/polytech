"""Integrer une EDP directement, une variable apres l'autre (syllabus, 9.1).

Quand l'equation ne porte que des derivees par rapport a une variable --
`u_xx = 0`, `u_y = f(y)` -- on l'integre comme une equation ordinaire.
Mais la « constante » d'integration ne depend pas de la variable
d'integration : c'est une **fonction arbitraire** des autres variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence


def par_integrations(
    membre: sp.Expr, integrations: Sequence[sp.Symbol], variables: Sequence[sp.Symbol]
) -> sp.Expr:
    """`u`, de `d^k u / d v_1 ... d v_k = membre`, une integration par variable.

    Chaque integration ajoute une fonction `F_i` des autres variables ; leur
    ordre est indifferent, les derivees croisees commutant (Schwarz).
    """
    if not integrations:
        msg = "aucune integration demandee"
        raise ValueError(msg)
    u = sp.sympify(membre)
    for rang, variable in enumerate(integrations, start=1):
        if variable not in variables:
            msg = f"{variable} n'est pas une variable de l'equation : {variables}"
            raise ValueError(msg)
        autres = [v for v in variables if v != variable]
        u = sp.integrate(u, variable) + sp.Function(f"F{rang}")(*autres)
    return u
