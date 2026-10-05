"""Les elements finis triangulaires : fonctions de forme et rigidite.

Syllabus de thermique, tome 1, premiere partie, section 5.3. Le domaine
est decoupe en elements, et le champ inconnu remplace par ses valeurs
nodales : `t = somme t_i N_i`. Chaque fonction de forme `N_i` vaut un a
son noeud et zero aux autres, et leur somme vaut un dans l'element.

Pour un triangle a trois noeuds elles sont lineaires, `a x + b y + c` ;
a six noeuds -- les sommets et les milieux des cotes --, quadratiques.
Leurs coefficients sortent du systeme lineaire qui ecrit ces valeurs
aux noeuds.

La methode de Galerkin pondere le residu de l'equation de
Fourier-Kirchhoff par ces memes fonctions. Sa forme faible donne, pour
chaque element, la matrice de rigidite thermique
`K_ij = integrale de lambda grad N_i . grad N_j` ; dans un triangle
lineaire les gradients sont constants, et l'integrale est leur produit
par l'aire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

X, Y = sp.symbols("x y")
MONOMES = {3: (X, Y, 1), 6: (X**2, X * Y, Y**2, X, Y, 1)}
"""La base polynomiale selon le nombre de noeuds de l'element."""

type Noeud = tuple[sp.Expr, sp.Expr]


def fonctions_de_forme(noeuds: Sequence[Noeud]) -> list[sp.Expr]:
    """Les `N_i`, telles que `N_i(x_j) = delta_ij`."""
    if len(noeuds) not in MONOMES:
        msg = f"trois ou six noeuds attendus, et non {len(noeuds)}"
        raise ValueError(msg)
    base = MONOMES[len(noeuds)]
    aux_noeuds = sp.Matrix(
        [[sp.sympify(m).subs({X: x, Y: y}) for m in base] for x, y in noeuds]
    )
    coefficients = aux_noeuds.inv()
    return [
        sp.expand(sum(coefficients[k, i] * m for k, m in enumerate(base)))
        for i in range(len(noeuds))
    ]


def _aire(sommets: Sequence[Noeud]) -> sp.Expr:
    (x1, y1), (x2, y2), (x3, y3) = sommets
    return sp.Abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2


def rigidite_d_un_triangle(
    sommets: Sequence[Noeud], *, conductivite: sp.Expr
) -> sp.Matrix:
    """`K_ij = lambda A grad N_i . grad N_j`, pour un triangle lineaire."""
    formes = fonctions_de_forme(sommets)
    gradients = [(sp.diff(f, X), sp.diff(f, Y)) for f in formes]
    aire = _aire(sommets)
    return sp.Matrix(
        3,
        3,
        lambda i, j: (
            conductivite
            * aire
            * (gradients[i][0] * gradients[j][0] + gradients[i][1] * gradients[j][1])
        ),
    )
