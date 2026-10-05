"""La ligne de temps : des particules marquees au meme instant.

Chapitre III.3 des planches, apres la ligne de courant, la trajectoire et
la ligne d'emission. Un fil tendu en travers de l'ecoulement marque d'un
coup toutes les particules qui le touchent -- par une impulsion
electrique, qui y degage des bulles d'hydrogene. La courbe qu'elles
forment ensuite est la ligne de temps : dans un ecoulement parallele,
elle prend la forme du profil de vitesse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.fluides.domain.cinematique.lignes import trajectoire

if TYPE_CHECKING:
    import sympy as sp


def ligne_de_temps(
    u: sp.Expr, v: sp.Expr, marquage: tuple[sp.Expr, sp.Expr]
) -> tuple[sp.Expr, sp.Expr]:
    """La position, a l'instant `t`, des particules marquees a l'instant zero.

    `marquage` est la courbe parametree du fil : chaque valeur du
    parametre designe une particule, que l'on suit le long de sa
    trajectoire.
    """
    return trajectoire(u, v, marquage[0], marquage[1])
