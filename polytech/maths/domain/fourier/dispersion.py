"""La dispersion d'une densite : son ecart-type.

Le syllabus (chapitre 10) appelle « variance » ce qui est l'ecart-type de la
gaussienne `e^(-a^2 t^2)` (`1 / (a racine 2)`) et de son image (`a racine 2`).
"""

from __future__ import annotations

import sympy as sp


def ecart_type(densite: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """`racine(int x^2 f / int f)` sur la droite ; leve si la masse est nulle."""
    masse = sp.integrate(densite, (variable, -sp.oo, sp.oo))
    if masse == 0:
        msg = "masse nulle : l'ecart-type n'est pas defini"
        raise ValueError(msg)
    second = sp.integrate(variable**2 * densite, (variable, -sp.oo, sp.oo))
    return sp.sqrt(sp.simplify(second / masse))
