"""Les produits scalaires du cours, la norme et l'orthogonalite qui en decoulent.

Le syllabus d'algebre (IX.1 a IX.5) : un produit scalaire est une
application `<. | .>` positive, lineaire a gauche et symetrique --
hermitienne, `<X | Y> = conj(<Y | X>)`, dans un espace complexe. Il en
emploie quatre :

* dans `R^n` et `C^n`, `<X | Y> = x_1 conj(y_1) + ... + x_n conj(y_n)` ;
* dans un espace de matrices reelles, `<A | B> = tr(A B^t)` ;
* dans un espace de polynomes ou de fonctions, l'integrale de `p q` sur
  un intervalle ;
* dans un espace de polynomes, la somme des `p(x_i) q(x_i)` en des points.

« Une fois le produit scalaire choisi », la norme est `racine(<X | X>)`,
la distance de `X` a `Y` est la norme de `X - Y`, et une famille est
orthogonale quand ses vecteurs le sont deux a deux.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    Produit = Callable[[Any, Any], sp.Expr]


def usuel(premier: sp.Matrix, second: sp.Matrix) -> sp.Expr:
    """`somme(x_i conj(y_i))`, dans `R^n` comme dans `C^n`."""
    termes = (x * sp.conjugate(y) for x, y in zip(premier, second, strict=True))
    return sp.simplify(sum(termes, sp.Integer(0)))


def de_la_trace(premiere: sp.Matrix, seconde: sp.Matrix) -> sp.Expr:
    """`tr(A B^t)` : la somme des produits des elements de meme place."""
    return (premiere * seconde.T).trace()


def integral(variable: sp.Symbol, debut: object, fin: object) -> Produit:
    """Le produit `(f, g) -> integrale de f g` sur `[debut, fin]`."""

    def produit(f: sp.Expr, g: sp.Expr) -> sp.Expr:
        return sp.simplify(sp.integrate(f * g, (variable, debut, fin)))

    return produit


def aux_points(variable: sp.Symbol, points: Sequence[object]) -> Produit:
    """Le produit `(p, q) -> somme des p(x_i) q(x_i)`."""

    def produit(p: sp.Expr, q: sp.Expr) -> sp.Expr:
        valeurs = (sp.sympify(p * q).subs(variable, point) for point in points)
        return sum(valeurs, sp.Integer(0))

    return produit


def norme(vecteur: Any, produit: Produit) -> sp.Expr:  # noqa: ANN401 -- tout element
    """`racine(<X | X>)`."""
    return sp.simplify(sp.sqrt(produit(vecteur, vecteur)))


def est_orthogonale(famille: Sequence[Any], produit: Produit) -> bool:
    """Vrai si les vecteurs sont orthogonaux deux a deux."""
    return all(
        sp.simplify(produit(premier, second)) == 0
        for premier, second in combinations(famille, 2)
    )
