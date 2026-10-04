"""Les conditions d'existence de la transformee de Laplace (planches 19 a 23).

Deux fonctions posent probleme : `1/t`, qui n'est pas integrable pres de
0, et `e^(t^2)`, qui croit « trop vite » pour que `e^(-pt)` l'ecrase.
D'ou deux conditions suffisantes :

* **CE1** -- `x` est integrable sur tout `[0, A]` ;
* **CE2** -- `x` est de **type exponentiel** d'ordre `alpha` :
  `e^(-alpha t) |x(t)|` tend vers 0, c'est-a-dire que `|x|` croit au pire
  comme une exponentielle.

Alors la transformee existe dans le demi-plan `Re(p) > alpha`.
"""

from __future__ import annotations

import sympy as sp
from sympy.calculus.util import continuous_domain

from polytech.maths.domain.symboles import TEMPS

DEMI_DROITE = sp.Interval(0, sp.oo)


def est_localement_integrable(original: sp.Expr) -> bool:
    """CE1 : continue sur `[0, +oo[`, ou d'integrale finie pres de ses singularites."""
    continu = continuous_domain(original, TEMPS, DEMI_DROITE)
    singularites = DEMI_DROITE - continu
    if not isinstance(singularites, sp.FiniteSet):
        return bool(singularites == sp.S.EmptySet)
    return all(
        sp.integrate(sp.Abs(original), (TEMPS, point, point + 1)).is_finite
        for point in singularites
    )


def est_de_type_exponentiel(original: sp.Expr, ordre: object) -> bool:
    """CE2 : `lim e^(-alpha t) |x(t)| = 0` quand `t` tend vers l'infini."""
    amortie = sp.exp(-sp.sympify(ordre) * TEMPS) * sp.Abs(original)
    return bool(sp.limit(amortie, TEMPS, sp.oo) == 0)


def verifie_les_conditions_d_existence(original: sp.Expr, ordre: object) -> bool:
    """CE1 et CE2 : la transformee existe alors pour `Re(p) > alpha`."""
    integrable = est_localement_integrable(original)
    return integrable and est_de_type_exponentiel(original, ordre)
