"""Les trois proprietes d'un produit scalaire reel (syllabus d'algebre, IX.1).

1) **positivite** : `<X|X> >= 0`, et `<X|X> = 0` seulement pour `X = 0` --
   la forme quadratique `X -> <X|X>` est definie positive ;
2) **linearite** : `<a1 X1 + a2 X2 | Y> = a1 <X1|Y> + a2 <X2|Y>` ;
3) **symetrie** : `<X|Y> = <Y|X>`.

Chaque propriete se verifie sur des elements generiques, dont les
composantes sont des symboles reels : `generique(nom)` en fabrique un.
Pour un espace de dimension finie, la positivite se lit sur la hessienne de
`<X|X>` par rapport aux composantes de `X` : homogene et de hessienne
definie positive, `<X|X>` est nul en `0` et seulement la. Un `<X|X>` qui
n'est pas polynomial est hors de portee : la fonction leve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable

    from polytech.maths.domain.algebre.produits_scalaires import Produit


def _nul(expression: sp.Expr) -> bool:
    return bool(sp.expand(expression) == 0)


def axiomes_du_produit_scalaire(
    produit: Produit, generique: Callable[[str], Any]
) -> dict[str, bool]:
    """Pour chaque propriete du cours, vrai si le produit la verifie."""
    x, x2, y = generique("x"), generique("u"), generique("y")
    a1, a2 = sp.symbols("a1 a2", real=True)
    carre = sp.expand(produit(x, x))
    composantes = sorted(sp.Matrix(x).free_symbols, key=str)
    if not carre.is_polynomial(*composantes):
        msg = f"<X|X> = {carre} n'est pas polynomial : positivite hors de portee"
        raise ValueError(msg)
    hessienne = sp.hessian(carre, composantes)
    positivite = bool(
        sp.Poly(carre, *composantes).is_homogeneous and hessienne.is_positive_definite
    )
    lineaire = produit(a1 * x + a2 * x2, y) - a1 * produit(x, y) - a2 * produit(x2, y)
    return {
        "positivite": positivite,
        "linearite": _nul(lineaire),
        "symetrie": _nul(produit(x, y) - produit(y, x)),
    }
