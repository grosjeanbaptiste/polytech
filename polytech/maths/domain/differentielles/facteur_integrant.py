"""Le facteur integrant : rendre exacte une equation qui ne l'est pas.

`P dx + Q dy = 0` n'est exacte que si `dP/dy = dQ/dx`. Mais `P` et `Q` ne
sont determines qu'a un facteur pres : multiplier par `h(x, y)` ne change
pas l'equation, et peut la rendre exacte. `h` est un **facteur
integrant** si `d(Qh)/dx = d(Ph)/dy`.

Le trouver en general n'a rien d'evident. Deux cas simples :

* si `(dP/dy - dQ/dx) / Q` ne depend que de `x`, alors `h = h(x)` avec
  `dh/h = (dP/dy - dQ/dx) / Q dx` ;
* si `(dP/dy - dQ/dx) / P` ne depend que de `y`, alors `h = h(y)` avec
  `dh/h = -(dP/dy - dQ/dx) / P dy`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE


def facteur_integrant(forme: FormeDifferentielle) -> sp.Expr:
    """`h(x)` ou `h(y)` ; leve si aucun des deux cas simples ne se presente."""
    ecart = forme.ecart_de_schwarz()
    if ecart == 0:
        return sp.Integer(1)
    en_x = sp.simplify(ecart / forme.facteur_en_y)
    if not en_x.has(INCONNUE):
        return sp.simplify(sp.exp(sp.integrate(en_x, VARIABLE)))
    en_y = sp.simplify(-ecart / forme.facteur_en_x)
    if not en_y.has(VARIABLE):
        return sp.simplify(sp.exp(sp.integrate(en_y, INCONNUE)))
    msg = "aucun facteur integrant fonction ni de x seul ni de y seul"
    raise ValueError(msg)


def rendre_exacte(forme: FormeDifferentielle) -> FormeDifferentielle:
    """La meme equation, multipliee par son facteur integrant."""
    h = facteur_integrant(forme)
    return FormeDifferentielle(
        sp.expand(forme.facteur_en_x * h), sp.expand(forme.facteur_en_y * h)
    )
