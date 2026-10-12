"""Le produit vectoriel et le produit mixte de l'espace a trois dimensions.

Dans une base cartesienne positivement orientee,

    a ^ b = (a_y b_z - a_z b_y, a_z b_x - a_x b_z, a_x b_y - a_y b_x).

Il est nul si les deux vecteurs sont paralleles, anticommutatif, et
distributif sur l'addition. Le **produit mixte** `a . (b ^ c)` est le
volume signe du parallelepipede bati sur les trois vecteurs.

C'est de la mathematique : la mecanique s'en sert pour le moment d'une
force, l'electromagnetisme pour la force de Laplace.

Sources : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
annexe A, pages 148 et 152 ; syllabus d'introduction aux sciences de
l'ingenieur, chapitre N.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

type Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]

DIMENSION = 3


def _de_l_espace(vecteur: Sequence[object]) -> Vecteur:
    if len(vecteur) != DIMENSION:
        msg = f"trois composantes attendues, et non {len(vecteur)}"
        raise ValueError(msg)
    x, y, z = (sp.sympify(composante) for composante in vecteur)
    return x, y, z


def produit_vectoriel(a: Sequence[object], b: Sequence[object]) -> Vecteur:
    """`a ^ b`, par ses composantes dans une base positivement orientee."""
    ax, ay, az = _de_l_espace(a)
    bx, by, bz = _de_l_espace(b)
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def matrice_antisymetrique(a: Sequence[object]) -> sp.Matrix:
    """`[a~]`, telle que `[a~] b = a ^ b`."""
    ax, ay, az = _de_l_espace(a)
    return sp.Matrix([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])


def produit_mixte(
    a: Sequence[object], b: Sequence[object], c: Sequence[object]
) -> sp.Expr:
    """`a . (b ^ c)` : le volume signe du parallelepipede."""
    termes = zip(_de_l_espace(a), produit_vectoriel(b, c), strict=True)
    return sp.expand(sum((x * y for x, y in termes), sp.Integer(0)))
