"""L'horloge a eau : la forme du reservoir dont le niveau baisse uniformement.

Le debit sortant suit Torricelli, `dV/dt = -k racine(z)` avec
`k = A_e racine(2 g)` et `A_e` l'aire de l'orifice. Pour un reservoir de
revolution de generatrice `z = f(r)`, `dV = pi r^2 dz`, d'ou

    dz/dt = -k racine(z) / (pi r^2).

La baisse est uniforme si `racine(z)` croit comme `r^2` : `z = b^2 r^4`,
et alors `dz/dt = -k b / pi = -r_e^2 b racine(2 g)`. Exiger une baisse
uniforme **impose** la forme.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.fondamentales.bernoulli import GRAVITE_EXACTE


def baisse_du_niveau(profil: sp.Expr, rayon: sp.Symbol, debit: object) -> sp.Expr:
    """`dz/dt = -k racine(f(r)) / (pi r^2)`, pour la generatrice `z = f(r)`."""
    return sp.simplify(-sp.sympify(debit) * sp.sqrt(profil) / (sp.pi * rayon**2))


def coefficient_du_profil(hauteur: object, rayon: object) -> sp.Expr:
    """`b`, tel que `z_0 = b^2 r_0^4` au bord superieur."""
    return sp.sqrt(sp.sympify(hauteur)) / sp.sympify(rayon) ** 2


def rayon_de_l_orifice(coefficient: object, vitesse: object) -> sp.Expr:
    """`r_e`, pour que le niveau baisse de `vitesse` : `r_e^2 b racine(2 g)`."""
    par_aire = sp.sympify(coefficient) * sp.sqrt(2 * GRAVITE_EXACTE)
    return sp.sqrt(sp.sympify(vitesse) / par_aire)
