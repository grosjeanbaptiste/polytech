"""Gram-Schmidt, projection orthogonale, distance et sous-espace orthogonal.

Le syllabus d'algebre (IX.1), pour un produit scalaire quelconque :

* le **procede de Gram-Schmidt** tire d'une famille libre `{X_1, ..., X_p}`
  une famille orthogonale, `v_i = X_i - somme(<X_i | v_l>/<v_l | v_l> v_l)` ;
  normer chaque `v_i` la rend orthonormale ;
* la **projection orthogonale** de `X` sur `W`, de base orthogonale
  `{v_1, ..., v_k}`, est `somme(<X | v_i>/<v_i | v_i> v_i)` : c'est la
  meilleure approximation de `X` dans `W`, et `d(X, W) = ||X - proj_W X||` ;
* le **sous-espace orthogonal** `W^perp` reunit les `X` orthogonaux a
  chaque generateur de `W`, et `dim W^perp = dim V - dim W`.

Les « vecteurs » sont des colonnes, des matrices, des polynomes ou des
fonctions : il suffit qu'ils s'ajoutent et se multiplient par un scalaire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.maths.domain.algebre.produits_scalaires import est_orthogonale, norme
from polytech.maths.domain.algebre.sous_espaces import base_du_noyau

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.maths.domain.algebre.produits_scalaires import Produit


def _est_nul(vecteur: Any) -> bool:  # noqa: ANN401 -- colonne, matrice ou expression
    if isinstance(vecteur, sp.MatrixBase):
        return bool(vecteur.is_zero_matrix)
    return bool(vecteur == 0)


def gram_schmidt(famille: Sequence[Any], produit: Produit) -> tuple[Any, ...]:
    """La famille orthogonale que le procede tire d'une famille libre."""
    orthogonale: list[Any] = []
    for vecteur in famille:
        reste = vecteur
        for retenu in orthogonale:
            reste = reste - produit(vecteur, retenu) / produit(retenu, retenu) * retenu
        reste = sp.simplify(reste)
        if _est_nul(reste):
            msg = "famille liee : un vecteur est combinaison des precedents"
            raise ValueError(msg)
        orthogonale.append(reste)
    return tuple(orthogonale)


def normaliser(famille: Sequence[Any], produit: Produit) -> tuple[Any, ...]:
    """Chaque vecteur divise par sa norme."""
    return tuple(sp.simplify(vecteur / norme(vecteur, produit)) for vecteur in famille)


def projection(
    vecteur: Any,  # noqa: ANN401 -- tout element de l'espace
    base: Sequence[Any],
    produit: Produit,
) -> Any:  # noqa: ANN401 -- un element du meme espace
    """`proj_W X`, pour une base orthogonale de `W`."""
    if not est_orthogonale(base, produit):
        msg = "base non orthogonale : la formule de projection ne vaut pas"
        raise ValueError(msg)
    projete = 0 * vecteur
    for retenu in base:
        projete = projete + produit(vecteur, retenu) / produit(retenu, retenu) * retenu
    return sp.simplify(projete)


def distance_au_sous_espace(
    vecteur: Any,  # noqa: ANN401 -- tout element de l'espace
    base: Sequence[Any],
    produit: Produit,
) -> sp.Expr:
    """`d(X, W) = ||X - proj_W X||`."""
    return norme(vecteur - projection(vecteur, base, produit), produit)


def orthogonal(
    generateurs: Sequence[Any], base: Sequence[Any], produit: Produit
) -> tuple[Any, ...]:
    """Une base de `W^perp`, les elements etant ecrits dans la base donnee de `V`."""
    conditions = sp.Matrix(
        [
            [produit(element, generateur) for element in base]
            for generateur in generateurs
        ]
    )
    return tuple(
        sp.simplify(
            sum((c * e for c, e in zip(coordonnees, base, strict=True)), 0 * base[0])
        )
        for coordonnees in base_du_noyau(conditions)
    )
