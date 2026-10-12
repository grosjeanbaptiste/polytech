"""Recueil d'algebre, chapitre VIII, section VIII.8 : les questions metriques.

Plusieurs sujets finissent par une « question relative au chapitre sur les
espaces metriques » : l'orthogonal du noyau, une projection. Le produit
scalaire est celui de la trace sur les matrices -- l'usuel, une fois la
matrice lue en vecteur -- et l'integrale sur `[-1, 1]` sur les polynomes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.application import recueil_algebre_ch08c_enonces as enonces
from polytech.maths.application import recueil_algebre_ch08c_etude as etude
from polytech.maths.domain.algebre import orthogonalisation as orth
from polytech.maths.domain.algebre import produits_scalaires as ps
from polytech.maths.domain.algebre.sous_espaces import base_du_noyau, somme

if TYPE_CHECKING:
    from polytech.maths.application.recueil_algebre_ch07_familles import Base

M = sp.Matrix
X = enonces.X
LEGENDRE: Final = ps.integral(X, -1, 1)
MONOMES_2: Final = (sp.Integer(1), X, X**2)
P_33: Final = sp.Symbol("a") * X**2 + sp.Symbol("b") * X + sp.Symbol("c")
W_44: Final = M([[1, 0, -1, 0], [-1, -1, 2, 1]])
"""Les deux equations de `W` au 44 : `a0 - a2 = 0`, `-a1 + 2 a2 + a3 - a0 = 0`."""
W_45: Final = ((1, 1, 0), (0, 1, 2))
V_45: Final = M([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0], [1, 0, 0, 1]]).T
BASE_W_45: Final = M([[1, 1, -1], [1, 0, 1], [0, 3, 0]]).T


def orthogonal_du_noyau(numero: int, cle: str = etude.GENERAL) -> Base:
    """Pour le produit de la trace : l'orthogonal du noyau est l'espace des lignes."""
    lignes = etude.matrice(numero, cle)
    return tuple(tuple(v) for v in lignes.T.columnspace())


def exercice_33() -> tuple[tuple[sp.Expr, ...], sp.Expr, sp.Expr]:
    """L'orthogonal du noyau pour l'integrale, et les deux projections de `p`."""
    noyau = (1 + X**2,)
    orthogonal = orth.orthogonal(noyau, MONOMES_2, LEGENDRE)
    sur_le_noyau = orth.projection(P_33, noyau, LEGENDRE)
    reste = sp.collect(sp.expand(P_33 - sur_le_noyau), X)
    return orthogonal, sp.factor(sur_le_noyau), reste


def exercice_41() -> tuple[sp.Expr, ...]:
    """Les polynomes de `P_3` orthogonaux a `X`, pour l'integrale sur `[-1, 1]`."""
    monomes = (*MONOMES_2, X**3)
    return orth.orthogonal((X,), monomes, LEGENDRE)


def exercice_42() -> tuple[sp.Matrix, tuple[sp.Expr, ...], Base]:
    """La matrice de `g`, `f(1, 1, 2)`, et l'orthogonal du noyau de `f o g`."""
    g = M([[1, 0, 0, 1], [1, 0, 0, 1], [2, 0, 0, 2]])
    composee = etude.matrice(42) * g
    lignes = tuple(tuple(v) for v in composee.T.columnspace())
    return g, tuple(etude.matrice(42) * M([1, 1, 2])), lignes


def exercice_44() -> Base:
    """Une base de `W + Ker f`."""
    return somme(etude.noyau(44), base_du_noyau(W_44))


def exercice_45() -> tuple[bool, Base, sp.Matrix]:
    """`W = Im f` ?, les matrices envoyees dans `W`, et la matrice dans `v`, `w`."""
    f = etude.matrice(45)
    egalite = etude.rang(45) == len(W_45)
    conditions = M([list(e) for e in base_du_noyau(M(W_45))]) * f
    return egalite, base_du_noyau(conditions), BASE_W_45.inv() * f * V_45
