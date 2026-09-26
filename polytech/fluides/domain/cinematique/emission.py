"""Cours III.3 -- la ligne d'emission, troisieme famille de courbes.

C'est ce que montre un filet de colorant : le lieu, a l'instant ``t``, de
toutes les particules passees par un meme point source. Chacune a ete lachee
a un instant ``tau`` different, et se trouve donc a l'endroit ou la mene la
trajectoire apres une duree ``t - tau``. Si l'invariant de cette famille ne
depend plus de ``tau``, la ligne d'emission se confond avec la ligne de
courant -- ce que le corrige affirmait sans le verifier.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.lignes import trajectoire
from polytech.fluides.domain.cinematique.symboles import T

INSTANT_EMISSION = sp.Symbol("tau", nonnegative=True)
"""Instant auquel une particule donnee du filet a quitte la source."""


def ligne_emission(
    u: sp.Expr,
    v: sp.Expr,
    x0: sp.Expr | float,
    y0: sp.Expr | float,
) -> tuple[sp.Expr, sp.Expr]:
    """Position a l'instant ``t`` de la particule emise en ``(x0, y0)`` a ``tau``."""
    abscisse, ordonnee = trajectoire(u, v, x0, y0)
    duree = T - INSTANT_EMISSION
    return (
        sp.simplify(abscisse.subs(T, duree)),
        sp.simplify(ordonnee.subs(T, duree)),
    )


def invariant_de_la_ligne_emission(
    u: sp.Expr,
    v: sp.Expr,
    x0: sp.Expr | float,
    y0: sp.Expr | float,
) -> sp.Expr:
    """Produit ``x y`` le long du filet.

    S'il ne contient plus ``tau``, toutes les particules du filet sont sur la
    meme hyperbole : la ligne d'emission est alors la ligne de courant.
    """
    abscisse, ordonnee = ligne_emission(u, v, x0, y0)
    return sp.simplify(abscisse * ordonnee)
