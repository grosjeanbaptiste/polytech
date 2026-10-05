"""Le laplacien, le d'alembertien, et les equations de la physique mathematique.

`Delta u` est la somme des derivees secondes de `u` par rapport aux
variables d'espace. Avec lui s'ecrivent :

* l'equation de Laplace, `Delta u = 0`, dont les solutions sont les
  fonctions **harmoniques** ;
* l'equation des ondes, `Delta u = u_tt / c^2`, soit `[] u = 0` avec le
  d'alembertien `[] = Delta - (1/c^2) d^2/dt^2` ;
* l'equation de la chaleur (ou de diffusion), `Delta u = u_t / k` ;
* l'equation de Helmholtz, `Delta u + lambda u = 0`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

ORDONNEE: Final = sp.Symbol("y", real=True)
"""La seconde variable d'espace des equations a deux variables."""


def laplacien(fonction: sp.Expr, variables: Sequence[sp.Symbol]) -> sp.Expr:
    """`Delta u`, somme des derivees secondes par rapport aux variables d'espace."""
    return sp.expand(sum((sp.diff(fonction, v, 2) for v in variables), sp.Integer(0)))


def dalembertien(
    fonction: sp.Expr, espace: Sequence[sp.Symbol], temps: sp.Symbol, celerite: object
) -> sp.Expr:
    """`Delta u - u_tt / c^2`."""
    acceleration = sp.diff(fonction, temps, 2) / sp.sympify(celerite) ** 2
    return laplacien(fonction, espace) - acceleration


def est_harmonique(fonction: sp.Expr, variables: Sequence[sp.Symbol]) -> bool:
    """Vrai si `Delta u = 0`."""
    return bool(sp.simplify(laplacien(fonction, variables)) == 0)
