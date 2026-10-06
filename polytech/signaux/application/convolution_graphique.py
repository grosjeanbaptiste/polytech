"""La note « Convolution graphique » : deux creneaux que l'on fait glisser.

`x` vaut `1/2` sur `[0, 1]` ; `h` vaut `+1` sur `[0, 1]` puis `-1` sur
`[1, 2]`. Leur convolution est l'aire du produit de `x(tau)` par
`h(t - tau)`, a mesure que `h` retourne glisse le long de l'axe.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import sympy as sp

from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.systemes import convolution

HAUTEUR: Final = sp.Rational(1, 2)
"""La hauteur du creneau `x`."""
ENTREE: Final = HAUTEUR * (sp.Heaviside(T) - sp.Heaviside(T - 1))
REPONSE: Final = sp.Heaviside(T) - 2 * sp.Heaviside(T - 1) + sp.Heaviside(T - 2)


def sortie(instant: object) -> sp.Expr:
    """`y(t) = (x * h)(t)`, par l'integrale de convolution."""
    valeur = sp.sympify(instant)
    if valeur <= 0:
        return sp.Integer(0)
    return sp.nsimplify(convolution(ENTREE, REPONSE).subs(T, valeur))


def support() -> tuple[int, int]:
    """L'intervalle hors duquel `y` est nulle : la somme des deux durees."""
    return 0, 1 + 2


def morceaux(variable: sp.Symbol) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """L'expression de `y` sur `[0, 1]`, `[1, 2]` puis `[2, 3]`."""
    monte = HAUTEUR * variable
    bascule = HAUTEUR * (1 - (variable - 1)) - HAUTEUR * (variable - 1)
    remonte = -HAUTEUR * (3 - variable)
    return sp.expand(monte), sp.expand(bascule), sp.expand(remonte)


def _recouvrement(debut: float, fin: float) -> float:
    """La longueur de `[debut, fin]` qui tombe dans `[0, 1]`, support de `x`."""
    return max(0.0, min(fin, 1.0) - max(debut, 0.0))


def aire_du_produit(instant: float) -> float:
    """La surface de `x(tau) h(t - tau)` : le lobe positif, puis le lobe negatif."""
    positif = _recouvrement(instant - 1, instant)
    negatif = _recouvrement(instant - 2, instant - 1)
    return float(HAUTEUR) * (positif - negatif)


def par_echantillonnage(*, pas: float) -> tuple[np.ndarray, np.ndarray]:
    """Les instants et `y`, par `numpy.convolve` sur les signaux echantillonnes."""
    x = np.full(round(1 / pas), float(HAUTEUR))
    h = np.concatenate([np.ones(round(1 / pas)), -np.ones(round(1 / pas))])
    y = np.convolve(x, h) * pas
    return np.arange(len(y)) * pas, y


def aire_de_la_sortie() -> sp.Expr:
    """L'integrale de `y` sur son support."""
    t = sp.Symbol("t")
    bornes = ((0, 1), (1, 2), (2, 3))
    return sum(
        (
            sp.integrate(m, (t, a, b))
            for m, (a, b) in zip(morceaux(t), bornes, strict=True)
        ),
        sp.Integer(0),
    )
