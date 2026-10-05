"""La matrice d'une application lineaire dans deux bases, et le changement de base.

Le syllabus d'algebre (VIII.5, VIII.6) : la matrice associee a `f` dans
les bases `v` et `w` a pour `j`-ieme colonne les coordonnees de `f(v_j)`
dans `w`, de sorte que `A_(v,w) X_v = f(X)_w`.

La **matrice de passage** `P_(v,v')` est celle de l'identite, `v` au
depart et `v'` a l'arrivee : sa `j`-ieme colonne porte les coordonnees de
`v_j` dans `v'`, et `X_(v') = P_(v,v') X_v`. D'ou

    A_(v,w) = P_(e',w) A_(e,e') P_(v,e).

Avec `V` et `W` les matrices dont les colonnes sont les vecteurs des deux
bases, ecrits dans les bases canoniques : `P_(v,e) = V`,
`P_(e',w) = W^-1`, et `A_(v,w) = W^-1 A V`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.maths.domain.algebre.familles_de_vecteurs import (
    est_une_base,
    matrice_des_lignes,
)

if TYPE_CHECKING:
    import sympy as sp

    from polytech.maths.domain.algebre.familles_de_vecteurs import Famille


def _colonnes(base: Famille, role: str) -> sp.Matrix:
    if not est_une_base(base):
        msg = f"la famille {role} n'est pas une base"
        raise ValueError(msg)
    return matrice_des_lignes(base).T


def matrice_de_passage(depart: Famille, arrivee: Famille) -> sp.Matrix:
    """`P_(v,v')` : les coordonnees dans `v` deviennent des coordonnees dans `v'`."""
    return _colonnes(arrivee, "d'arrivee").inv() * _colonnes(depart, "de depart")


def matrice_associee(
    canonique: sp.Matrix, depart: Famille, arrivee: Famille
) -> sp.Matrix:
    """`A_(v,w) = P_(e',w) A_(e,e') P_(v,e)`."""
    passage = _colonnes(arrivee, "d'arrivee").inv()
    return passage * canonique * _colonnes(depart, "de depart")


def matrice_canonique(
    associee: sp.Matrix, depart: Famille, arrivee: Famille
) -> sp.Matrix:
    """`A_(e,e') = P_(w,e') A_(v,w) P_(e,v)` : le chemin inverse."""
    passage = _colonnes(depart, "de depart").inv()
    return _colonnes(arrivee, "d'arrivee") * associee * passage
