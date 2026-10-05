"""Le deplacement d'un coin de particule, decompose en quatre mouvements.

Chapitre III.4 des planches. Une particule carree `ABCD`, de cotes `dx`
et `dy`, devient `A'B'C'D'` apres `dt`. Les coordonnees du coin oppose
`D'` s'ecrivent, au premier ordre,

`x_D' = x + dx + u dt + (du/dx) dx dt + (1/2)(du/dy + dv/dx) dy dt
        - (1/2)(dv/dx - du/dy) dy dt`,

et de meme pour `y_D'` : une **translation** a la vitesse du coin `A`,
une **dilatation** le long de chaque cote, un **cisaillement** qui
referme l'angle droit, une **rotation** en bloc. Leur somme est le
deplacement de `D`, sa vitesse fois `dt`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.operateurs import (
    taux_de_rotation,
    tenseur_taux_deformation,
)

Deplacement = tuple[sp.Expr, sp.Expr]


def contributions_au_deplacement(
    u: sp.Expr, v: sp.Expr, cotes: tuple[sp.Expr, sp.Expr], duree: sp.Expr
) -> dict[str, Deplacement]:
    """Les quatre parts du deplacement du coin `D`, chacune en `(dx, dy)`."""
    dx, dy = cotes
    deformation = tenseur_taux_deformation(u, v)
    rotation = taux_de_rotation(u, v)
    return {
        "translation": (u * duree, v * duree),
        "dilatation": (
            deformation[0, 0] * dx * duree,
            deformation[1, 1] * dy * duree,
        ),
        "cisaillement": (
            deformation[0, 1] * dy * duree,
            deformation[1, 0] * dx * duree,
        ),
        "rotation": (-rotation * dy * duree, rotation * dx * duree),
    }
